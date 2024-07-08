# Load standard modules
import numpy as np

import matplotlib
from matplotlib import pyplot as plt

# Load tudatpy modules
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.kernel.interface.spice import load_kernel
from tudatpy.numerical_simulation import environment
from tudatpy.numerical_simulation import environment_setup, propagation_setup
from tudatpy.astro import element_conversion
from tudatpy import constants
from tudatpy.util import result2array
from tudatpy.astro.time_conversion import DateTime

## Configuration

# Load spice kernels
spice.load_standard_kernels()
spice.get_total_count_of_kernels_loaded()
spice.load_kernel('C:\TudatProjects\OrbitIntegration\de438.bsp')
spice.load_kernel('C:\TudatProjects\OrbitIntegration\sat427.bsp')

# Set simulation start and end epochs
simulation_start_epoch = DateTime(2000, 1, 1).epoch()
simulation_end_epoch   = DateTime(2000, 1, 11, 1).epoch()

## Environment setup
"""
Create the environment for our simulation. This covers the creation of (celestial) bodies, vehicle(s), and environment interfaces.
"""

### Create the bodies

# Define string names for bodies to be created from default.
bodies_to_create = ["Enceladus", "Saturn", "Rhea", "Dione", "Titan", "Tethys", "Mimas", "Sun", "Jupiter", "Hyperion"]

# Use "Enceladus"/"J2000" as global frame origin and orientation.
global_frame_origin = "Enceladus"
global_frame_orientation = "IAU_Enceladus"#"ECLIPJ2000"

# Create default body settings, usually from `spice`.
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation)

# Define the spherical harmonics gravity model for Enceladus
gravitational_parameter = 7.211581150E+9
reference_radius = 256300.0

normalized_cosine_coefficients = [
    [1,                   0,                   0,                   0],
    [0,                   0,                   0,                   0],
#    [-2.449590E-03,       6.088329E-06,        2.351520E-03,        0], #Park2024
    [-2.4306950E-03, 7.126289E-06, 2.400939E-03, 0], #Iess2014 normalized
#    [-5.4352E-03, 9.2E-06, 1.5498E-03, 0], #Iess2014 un-normalized
#    [-2.5E-03, 0, 2.5E-03, 0], #Russell2009 normalized
#    [6.720964E-05,        0,                   0,                   0] #Park2024
    [4.357930E-05, 0, 0, 0] #Iess2014 normalized
#    [1.153E-04, 0, 0, 0] #Iess2014 un-normalized
#    [1.0E-05, 0, 0, 0] #Russell2009 normalized
]
normalized_sine_coefficients = [
    [0,                   0,                   0,                   0],
    [0,                   0,                   0,                   0],
#    [0,                   5.886934E-06,        -4.265084E-04,       0], #Park2024
    [0, 3.082894E-05, 3.501176E-05, 0], #Iess2014 normalized
#    [0, 3.98E-05, 2.26E-05, 0], #Iess2014 un-normalized
#    [0, 0.0E-05, 0.0E-05, 0],  #Russell2009 normalized
    [0,                   0,                   0,                   0]
]

associated_reference_frame = "IAU_Enceladus"
# Create the gravity field settings and add them to the body "Enceladus"
body_settings.get( "Enceladus" ).gravity_field_settings = environment_setup.gravity_field.spherical_harmonic(
    gravitational_parameter,
    reference_radius,
    normalized_cosine_coefficients,
    normalized_sine_coefficients,
    associated_reference_frame )

# Add setting for moment of inertia
body_settings.get( "Enceladus" ).gravity_field_settings.scaled_mean_moment_of_inertia = 0.335

# Create system of selected celestial bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

### Create the vehicle
# Create vehicle objects.
bodies.create_empty_body("Orbiter")
bodies.get("Orbiter").mass = 2150

## Propagation setup

# Define bodies that are propagated
bodies_to_propagate = ["Orbiter"]

# Define central bodies of propagation
central_bodies = ["Enceladus"]

### Create the acceleration model

# Define accelerations acting on Orbiter by Sun, Saturn, and Enceladus.
accelerations_settings_orbiter = dict(
    Enceladus=[
        #propagation_setup.acceleration.point_mass_gravity()
        propagation_setup.acceleration.spherical_harmonic_gravity(3, 3)
    ],
    Saturn=[
        propagation_setup.acceleration.point_mass_gravity()
    ],
)
"""    Rhea=[
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Dione=[
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Titan=[
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Tethys=[
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Mimas=[
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Sun=[
        #    propagation_setup.acceleration.radiation_pressure(),
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Jupiter=[
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Hyperion = [
        propagation_setup.acceleration.point_mass_gravity()
    ],
)"""

# Create global accelerations settings dictionary.
acceleration_settings = {"Orbiter": accelerations_settings_orbiter}

# Create acceleration models.
acceleration_models = propagation_setup.create_acceleration_models(
    bodies,
    acceleration_settings,
    bodies_to_propagate,
    central_bodies)

### Define the initial state

#orbiter_ephemeris = environment.TleEphemeris( "Earth", "J2000", orbiter_tle, False )
#initial_state = orbiter_ephemeris.cartesian_state( simulation_start_epoch )

# Set initial conditions for the satellite that will be propagated in this simulation. The initial conditions
# are given in Cartesian elements (units: meters and meters per second)
enceladus_gravitational_parameter = bodies.get("Enceladus").gravitational_parameter
initial_state = [-235527.2394516388, -437772.5059013570, 0, 50.65296483365822, -37.64519699459160, 102.8014839043201]

# Set initial conditions for the satellite that will be propagated in this simulation. The initial conditions
# are given in Keplerian elements and later on converted to Cartesian elements
"""initial_state = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=enceladus_gravitational_parameter, #7.211581150E+9, #enceladus_gravitational_parameter, #7.210443E+9, # 7.211581150E+9
    #semi_major_axis=4.53620e+05,
    semi_major_axis=4.987636181497566e+05,
    #eccentricity=0.7594e-01,
    eccentricity=0.7594742316109369e-01,
    #inclination=55.17e+00,
    inclination=0.7,#1.0249445183128482739, #58.72499513454099e+00,
    #argument_of_periapsis=-0.9185596836014398e+02, #90.0e+00,
    argument_of_periapsis=-1.6031890854882719921, #-0.9185596836014398e+02,  #90.0e+00,
    #longitude_of_ascending_node=0.2417191769293457e+03,
    longitude_of_ascending_node=4.2187955026173344919, #0.2417191769293457e+03,
    #true_anomaly=0.9185596836014398e+02,
    true_anomaly=1.6031890854882719921, #0.9185596836014398e+02,
)"""

### Define dependent variables to save

# Define list of dependent variables to save
dependent_variables_to_save = ([
    propagation_setup.dependent_variable.total_acceleration("Orbiter"),
    propagation_setup.dependent_variable.keplerian_state("Orbiter", "Enceladus"),
    propagation_setup.dependent_variable.latitude("Orbiter", "Enceladus"),
    propagation_setup.dependent_variable.longitude("Orbiter", "Enceladus"),
    propagation_setup.dependent_variable.single_acceleration_norm(
        #propagation_setup.acceleration.point_mass_gravity_type, "Orbiter", "Enceladus"
        propagation_setup.acceleration.spherical_harmonic_gravity_type, "Orbiter", "Enceladus"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "Orbiter", "Saturn"
    )
])
""",
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "Orbiter", "Rhea"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "Orbiter", "Dione"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "Orbiter", "Titan"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "Orbiter", "Tethys"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "Orbiter", "Mimas"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "Orbiter", "Sun"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "Orbiter", "Jupiter"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "Orbiter", "Hyperion"
    )
]"""

### Create the propagator settings

# Create termination settings
termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)

# Create numerical integrator settings
fixed_step_size = 10.0
integrator_settings = propagation_setup.integrator.runge_kutta_4(fixed_step_size)

# Create propagation settings
propagator_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    initial_state,
    simulation_start_epoch,
    integrator_settings,
    termination_condition,
    output_variables=dependent_variables_to_save
)

## Propagate the orbit

# Create simulation object and propagate the dynamics
dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    bodies, propagator_settings
)

# Extract the resulting state and depedent variable history and convert it to an ndarray
states = dynamics_simulator.state_history
states_array = result2array(states)
dep_vars = dynamics_simulator.dependent_variable_history
dep_vars_array = result2array(dep_vars)

## Post-process the propagation results
"""
The results of the propagation are then processed to a more user-friendly form.
"""

### Total acceleration over time

# Plot total acceleration as function of time
time_hours = dep_vars_array[:,0]/3600
total_acceleration_norm = np.linalg.norm(dep_vars_array[:,1:4], axis=1)
plt.figure(figsize=(9, 5))
plt.title("Total acceleration norm on Orbiter over the course of propagation.")
plt.plot(time_hours, total_acceleration_norm, linewidth=0.5)
plt.xlabel('Time [hr]')
plt.ylabel('Total Acceleration [m/s$^2$]')
plt.xlim([min(time_hours), max(time_hours)])
plt.grid()
plt.tight_layout()
plt.show()

### Ground track

# Plot ground track for a period of 3 hours
latitude = dep_vars_array[:,10]
longitude = dep_vars_array[:,11]
hours = 270
subset = int(len(time_hours) / 24 * hours)
latitude = np.rad2deg(latitude[0: subset])
longitude = np.rad2deg(longitude[0: subset])
plt.figure(figsize=(9, 5))
plt.title("3 hour ground track of Orbiter")
#plt.plot(longitude,latitude, linewidth=0.5)
plt.scatter(longitude, latitude, marker='.', s=0.05)
plt.xlabel('Longitude [deg]')
plt.ylabel('Latitude [deg]')
plt.xlim([min(longitude), max(longitude)])
plt.yticks(np.arange(-90, 91, step=45))
plt.grid()
plt.tight_layout()
plt.show()

### Kepler elements over time
"""
Let's now plot each of the 6 Kepler element as a function of time, also as saved in the dependent variables.
"""

# Plot Kepler elements as a function of time
kepler_elements = dep_vars_array[:,4:10]
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(9, 12))
fig.suptitle('Evolution of Kepler elements over the course of the propagation.')

# Semi-major Axis
semi_major_axis = kepler_elements[:,0] / 1e3
ax1.plot(time_hours, semi_major_axis)
ax1.set_ylabel('Semi-major axis [km]')

# Eccentricity
eccentricity = kepler_elements[:,1]
ax2.plot(time_hours, eccentricity)
ax2.set_ylabel('Eccentricity [-]')

# Inclination
inclination = np.rad2deg(kepler_elements[:,2])
ax3.plot(time_hours, inclination)
ax3.set_ylabel('Inclination [deg]')

# Argument of Periapsis
argument_of_periapsis = np.rad2deg(kepler_elements[:,3])
ax4.plot(time_hours, argument_of_periapsis)
ax4.set_ylabel('Argument of Periapsis [deg]')

# Right Ascension of the Ascending Node
raan = np.rad2deg(kepler_elements[:,4])
ax5.plot(time_hours, raan)
ax5.set_ylabel('RAAN [deg]')

# True Anomaly
true_anomaly = np.rad2deg(kepler_elements[:,5])
ax6.scatter(time_hours, true_anomaly, s=1)
ax6.set_ylabel('True Anomaly [deg]')
ax6.set_yticks(np.arange(0, 361, step=60))

for ax in fig.get_axes():
    ax.set_xlabel('Time [hr]')
    ax.set_xlim([min(time_hours), max(time_hours)])
    ax.grid()
ax1.set_ylim([450, 550])
#ax1.set_xlim([0, 20])
plt.tight_layout()
plt.show()

### Accelerations over time

plt.figure(figsize=(9, 5))

# Spherical Harmonics Gravity Acceleration Enceladus
acceleration_norm_sh_enceladus = dep_vars_array[:,12]
plt.plot(time_hours, acceleration_norm_sh_enceladus, label='SH Enceladus')

# Point Mass Gravity Acceleration Saturn
acceleration_norm_pm_saturn = dep_vars_array[:,13]
plt.plot(time_hours, acceleration_norm_pm_saturn, label='PM Saturn')
"""
# Point Mass Gravity Acceleration Rhea
acceleration_norm_pm_sun = dep_vars_array[:,14]
plt.plot(time_hours, acceleration_norm_pm_sun, label='PM Rhea')

# Point Mass Gravity Acceleration Dione
acceleration_norm_pm_sun = dep_vars_array[:,15]
plt.plot(time_hours, acceleration_norm_pm_sun, label='PM Dione')

# Point Mass Gravity Acceleration Titan
acceleration_norm_pm_sun = dep_vars_array[:,16]
plt.plot(time_hours, acceleration_norm_pm_sun, label='PM Titan')

# Point Mass Gravity Acceleration Tethys
acceleration_norm_pm_sun = dep_vars_array[:,17]
plt.plot(time_hours, acceleration_norm_pm_sun, label='PM Tethys')

# Point Mass Gravity Acceleration Mimas
acceleration_norm_pm_sun = dep_vars_array[:,18]
plt.plot(time_hours, acceleration_norm_pm_sun, label='PM Mimas')

# Point Mass Gravity Acceleration Sun
acceleration_norm_pm_sun = dep_vars_array[:,19]
plt.plot(time_hours, acceleration_norm_pm_sun, label='PM Sun')

# Point Mass Gravity Acceleration Jupiter
acceleration_norm_pm_jupiter = dep_vars_array[:,20]
plt.plot(time_hours, acceleration_norm_pm_jupiter, label='PM Jupiter')

# Point Mass Gravity Acceleration Hyperion
acceleration_norm_pm_sun = dep_vars_array[:,21]
plt.plot(time_hours, acceleration_norm_pm_sun, label='PM Hyperion')
"""
# Cannonball Radiation Pressure Acceleration Sun
#acceleration_norm_rp_sun = dep_vars_array[:,18]
#plt.plot(time_hours, acceleration_norm_rp_sun, label='Radiation Pressure Sun')

plt.xlim([min(time_hours), max(time_hours)])
plt.xlabel('Time [hr]')
plt.ylabel('Acceleration Norm [m/s$^2$]')

plt.legend(bbox_to_anchor=(1.005, 1))
plt.suptitle("Accelerations norms on Orbiter, distinguished by type and origin, over the course of propagation.")
plt.yscale('log')
plt.grid()
plt.tight_layout()
plt.show()

print(
    f"""
Single Earth-Orbiting Satellite Example.
The initial position vector of Orbiter is [km]: \n{
    states[simulation_start_epoch][:3] / 1E3}
The initial velocity vector of Orbiter is [km/s]: \n{
    states[simulation_start_epoch][3:] / 1E3}
\nAfter {simulation_end_epoch} seconds the position vector of Orbiter is [km]: \n{
    states[simulation_end_epoch][:3] / 1E3}
And the velocity vector of Orbiter is [km/s]: \n{
    states[simulation_start_epoch][3:] / 1E3}
    """
)

# Define a 3D figure using pyplot
fig = plt.figure(figsize=(6,6), dpi=125)
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f'Orbiter trajectory around Enceladus')

# Plot the positional state history
ax.plot(states_array[:, 1], states_array[:, 2], states_array[:, 3], label=bodies_to_propagate[0], linestyle='-')
ax.scatter(0.0, 0.0, 0.0, label="Enceladus", marker='o', color='black')
ax.view_init(0, 45)

# Add the legend and labels, then show the plot
ax.legend()
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
plt.show()
