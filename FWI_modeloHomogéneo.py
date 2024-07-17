import numpy as np  
import matplotlib.pyplot as plt
from sympy import Min, Max

from devito import configuration, Function, norm, mmax, Eq, Operator

from examples.seismic import demo_model, plot_velocity, plot_image, plot_shotrecord, AcquisitionGeometry, Receiver
from examples.seismic.acoustic import AcousticWaveSolver

# Turn off logging
configuration['log-level'] = "ERROR"
# Setup
nshots = 9  # Numero de disparos para generar el gradiente
nreceivers = 70  # Numero de receptores por disparo
fwi_iterations = 170 # Número de iteraciones de FWI

# Definición del modelo verdadero y el modelo inicial
shape = (70, 70)  # Number of grid point (nx, nz)
spacing = (10., 10.)  # Grid spacing in m. The domain size is now 1km by 1km
origin = (0., 0.)  # Need origin to define relative source and receiver locations

model = demo_model('layers-isotropic', 
                   origin=origin, shape=shape, spacing=spacing, nbl=40)

model0 = demo_model('circle-isotropic', vp_circle=2.5, vp_background=2.5,
                    origin=origin, shape=shape, spacing=spacing, nbl=40,
                    grid=model.grid)

plot_velocity(model)
plot_velocity(model0)

t0 = 0. #tiempo inicial
tn = 1000. #tiempo final (ms)
tn2 = 1748.
f0 = 0.010 #frecuencia: 10 Hertzh
# First, position source centrally in all dimensions, then set depth
src_coordinates = np.empty((1, 2))
src_coordinates[:,0] = 350
src_coordinates[:,1] = 0  

# Define acquisition geometry: receivers

# Initialize receivers for synthetic and imaging data
rec_coordinates = np.empty((nreceivers, 2))
rec_coordinates[:, 0] = np.linspace(0, model.domain_size[0], num=nreceivers)
rec_coordinates[:, 1] =  0

# Geometry (ondícula Ricker)
geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates, t0, tn2, f0=f0, src_type='Ricker')
# We can plot the time signature to see the wavelet
geometry.src.show()

# Plot acquisition geometry
plot_velocity(model, source=geometry.src_positions,
              receiver=geometry.rec_positions[::4, :])

#Synthetic data with forward operator (acoustic solver)
solver = AcousticWaveSolver(model, geometry, space_order=4)
true_d, _, _ = solver.forward(vp=model.vp)

#Initial data with forward operator (acoustic solver)
smooth_d, _, _ = solver.forward(vp=model0.vp)

# Plot shot record for true and smooth velocity model and the difference
plot_shotrecord(true_d.data, model, t0, tn)
plot_shotrecord(smooth_d.data, model, t0, tn)
plot_shotrecord(smooth_d.data - true_d.data, model, t0, tn)


# Prepare the varying source locations sources (¿por qué varía?)
source_locations = np.empty((nshots, 2), dtype=np.float32)
source_locations[:, 0] = np.linspace(0, 1000, num=nshots)
source_locations[:, 1] = 0

plot_velocity(model, source=source_locations)

# Computes the residual between observed and synthetic data into the residual
def compute_residual(residual, dobs, dsyn):
    if residual.grid.distributor.is_parallel:
        # If we run with MPI, we have to compute the residual via an operator
        # First make sure we can take the difference and that receivers are at the 
        # same position
        assert np.allclose(dobs.coordinates.data[:], dsyn.coordinates.data)
        assert np.allclose(residual.coordinates.data[:], dsyn.coordinates.data)
        # Create a difference operator
        diff_eq = Eq(residual, dsyn.subs({dsyn.dimensions[-1]: residual.dimensions[-1]}) -
                               dobs.subs({dobs.dimensions[-1]: residual.dimensions[-1]}))
        Operator(diff_eq)()
    else:
        # A simple data difference is enough in serial
        residual.data[:] = dsyn.data[:] - dobs.data[:]
    
    return residual

# Create FWI gradient kernel 
def fwi_gradient(vp_in):    
    # Create symbols to hold the gradient
    grad = Function(name="grad", grid=model.grid)
    # Create placeholders for the data residual and data
    residual = Receiver(name='residual', grid=model.grid,
                        time_range=geometry.time_axis, 
                        coordinates=geometry.rec_positions)
    d_obs = Receiver(name='d_obs', grid=model.grid,
                     time_range=geometry.time_axis, 
                     coordinates=geometry.rec_positions)
    d_syn = Receiver(name='d_syn', grid=model.grid,
                     time_range=geometry.time_axis, 
                     coordinates=geometry.rec_positions)
    objective = 0.
    for i in range(nshots):
        # Se actualiza la posición de la fuente para cada iteración en el rango nshots
        geometry.src_positions[0,:] = source_locations[i,:]
        
        # Generar datos sintéticos
        _, _, _ = solver.forward(vp=model.vp, rec=d_obs)
        
        # Compute smooth data and full forward wavefield u0
        _, u0, _ = solver.forward(vp=vp_in, save=True, rec=d_syn)
        
        # Compute gradient from data residual and update objective function 
        compute_residual(residual, d_obs, d_syn)
        
        objective += .5*norm(residual)**2 #norma L2
        solver.gradient(rec=residual, u=u0, vp=vp_in, grad=grad)
    
    return objective, grad

# Compute gradient of initial model
ff, update = fwi_gradient(model0.vp)

# Plot the FWI gradient
plot_image(-update.data, vmin=-1e4, vmax=1e4, cmap="jet")

# Plot the difference between the true and initial model.
# This is not known in practice as only the initial model is provided.
plot_image(model0.vp.data - model.vp.data, vmin=-1e-1, vmax=1e-1, cmap="jet")

# Show what the update does to the model
alpha = .5 / mmax(update)
plot_image(model0.vp.data + alpha*update.data, vmin=2.5, vmax=3.0, cmap="jet")

# Define bounding box constraints on the solution.
def update_with_box(vp, alpha, dm, vmin=2.0, vmax=3.5):
    """
    Apply gradient update in-place to vp with box constraint

    Notes:
    ------
    For more advanced algorithm, one will need to gather the non-distributed
    velocity array to apply constrains and such.
    """
    update = vp + alpha * dm
    update_eq = Eq(vp, Max(Min(update, vmax), vmin))
    Operator(update_eq)()

# Run FWI with gradient descent
history = np.zeros((fwi_iterations, 1))
for i in range(0, fwi_iterations):
    # Compute the functional value and gradient for the current
    # model estimate
    phi, direction = fwi_gradient(model0.vp)
    
    # Store the history of the functional values
    history[i] = phi
    
    # Artificial Step length for gradient descent
    # In practice this would be replaced by a Linesearch (Wolfe, ...)
    # that would guarantee functional decrease Phi(m-alpha g) <= epsilon Phi(m)
    # where epsilon is a minimum decrease constant
    alpha = .05 / mmax(direction)
    
    # Update the model estimate and enforce minimum/maximum values
    update_with_box(model0.vp , alpha , direction)
    
    # Log the progress made
    print('La función de costo es %f en la iteración %d' % (phi, i+1))

# Modelo invertido
plot_velocity(model0)

# Decaimiento de la función de costo
plt.figure()
plt.loglog(history)
plt.xlabel('Numero de iteraciones')
plt.ylabel('Valor de la función de costo')
plt.title('Convergencia')
plt.show()

