# Load standard modules
import numpy as np
import math
import tudatpy

import matplotlib
from matplotlib import pyplot as plt

# Load tudatpy modules
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.kernel.interface.spice import load_kernel
from tudatpy.kernel.numerical_simulation.environment import RotationalEphemeris
from tudatpy.numerical_simulation import environment
from tudatpy.numerical_simulation import environment_setup, propagation_setup
from tudatpy.astro import element_conversion
from tudatpy import constants
from tudatpy.util import result2array
from tudatpy.astro.time_conversion import DateTime
from tudatpy import astro

## Configuration

# Load spice kernels
spice.load_standard_kernels()
spice.load_kernel('C:\TudatProjects\OrbitIntegration\de438.bsp')
spice.load_kernel('C:\TudatProjects\OrbitIntegration\sat427.bsp')
spice.load_kernel('C:\TudatProjects\OrbitIntegration\pck00010.tpc')
spice.get_total_count_of_kernels_loaded()

# Set simulation start and end epochs; this is one period of the K2 orbit from Benedikter et al. (2022)
simulation_start_epoch = DateTime(2000, 1, 1, 0, 0, 0).epoch()
simulation_end_epoch   = DateTime(2000, 1, 3, 13, 48).epoch() 

print(simulation_start_epoch)
print(simulation_end_epoch)

## Environment setup
"""
Create the environment for our simulation. This covers the creation of (celestial) bodies, vehicle(s), and environment interfaces.
"""

### Create the bodies

# Define string names for bodies to be created from default.
bodies_to_create = ["Enceladus", "Saturn"]

# Use "ECLIPJ2000" as global frame origin and orientation.
global_frame_origin = "Enceladus"
global_frame_orientation = "ECLIPJ2000"

# Create default body settings, usually from `spice`.
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation)

# Define the spherical harmonics gravity model for Enceladus
gravitational_parameter_enc = 7.211292085479989E+9
reference_radius_enc = 252240.0

# Normalize the spherical harmonic coefficients
nor_sh_enc=astro.gravitation.normalize_spherical_harmonic_coefficients(
    [ #Iess et al. 2014
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [5.4352E-03, 9.2E-06, 1.5498E-03, 0],
        [-1.15E-04, 0, 0, 0]
    ],
    [ #Iess et al. 2014
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 3.98E-05, 2.26E-05, 0],
        [0, 0, 0, 0]
    ])

# Assign normalized cosine and sine coefficients
normalized_cosine_coefficients = nor_sh_enc[0]
normalized_sine_coefficients = nor_sh_enc[1]

associated_reference_frame = "IAU_Enceladus"

# Create the gravity field settings and add them to the body "Enceladus"
body_settings.get( "Enceladus" ).gravity_field_settings = environment_setup.gravity_field.spherical_harmonic(
    gravitational_parameter_enc,
    reference_radius_enc,
    normalized_cosine_coefficients,
    normalized_sine_coefficients,
    associated_reference_frame )

# Add setting for moment of inertia
body_settings.get( "Enceladus" ).gravity_field_settings.scaled_mean_moment_of_inertia = 0.335

# Define the spherical harmonics gravity model for Saturn
saturn_gravitational_parameter = 3.7931208E+016
saturn_reference_radius = 60330000.0

# Normalize the spherical harmonic coefficients
nor_sh_sat=astro.gravitation.normalize_spherical_harmonic_coefficients(
    [ #Iess et al. 2019
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-16290.71E-6, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [935.83E-6, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-86.14E-6, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [10.E-6, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    [ #Iess et al. 2019
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

# Assign normalized cosine and sine coefficients
saturn_normalized_cosine_coefficients = nor_sh_sat[0]
saturn_normalized_sine_coefficients = nor_sh_sat[1]

saturn_associated_reference_frame = "IAU_Saturn"

# Create the gravity field settings and add them to the body "Saturn"
body_settings.get( "Saturn" ).gravity_field_settings = environment_setup.gravity_field.spherical_harmonic(
    saturn_gravitational_parameter,
    saturn_reference_radius,
    saturn_normalized_cosine_coefficients,
    saturn_normalized_sine_coefficients,
    saturn_associated_reference_frame )

# Add setting for moment of inertia for Saturn
body_settings.get("Saturn").gravity_field_settings.scaled_mean_moment_of_inertia = 0.210

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
        propagation_setup.acceleration.spherical_harmonic_gravity(8,8)
    ],
)

# Create global accelerations settings dictionary.
acceleration_settings = {"Orbiter": accelerations_settings_orbiter}

# Create acceleration models.
acceleration_models = propagation_setup.create_acceleration_models(
    bodies,
    acceleration_settings,
    bodies_to_propagate,
    central_bodies)

### Define the initial state
# Set initial conditions for the satellite that will be propagated in this simulation. The initial conditions
# are given in Cartesian elements (units: meters and meters per second)

enceladus_gravitational_parameter=gravitational_parameter_enc

# Set initial conditions for the satellite that will be propagated in this simulation. The initial conditions
# are given in Keplerian elements and later on converted to Cartesian elements

# Orbit K2 from Benedikter et al. (2022)
# initial elements [a [km], e, i [deg], Omega [deg], omega [deg], Ma [deg]] (the order is Omega, omega, not omega, Omega)
# sv_start = np.array([4.90400000e+02, 7.59474232e-02, 5.62500000e+01, 2.41719177e+02, 2.68144032e+02, 8.31499110e+01])
initial_state_enceladus_fixed_list = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=enceladus_gravitational_parameter,
    semi_major_axis=0.490400e+06,
    eccentricity=0.759474232e-01,
    inclination=np.deg2rad(5.62500000e+01),
    argument_of_periapsis=np.deg2rad(2.68144032e+02),
    longitude_of_ascending_node=np.deg2rad(2.41719177e+02),
    true_anomaly=np.deg2rad(8.31499110e+01),
)

# Assign initial state in Cartesian coordinates in body-fixed frame
initial_state_enceladus_fixed = np.array(initial_state_enceladus_fixed_list)

# Get rotation matrices between IAU_Enceladus and global_frame_orientation
rotation_matrix = spice.compute_rotation_matrix_between_frames("IAU_Enceladus",global_frame_orientation, simulation_start_epoch )
rotation_matrix_back = spice.compute_rotation_matrix_between_frames(global_frame_orientation,"IAU_Enceladus", simulation_start_epoch )

# Assign initial state in Cartesian coordinates in inertial frame
initial_state = np.ndarray([6])
initial_state[0:3] = rotation_matrix.dot(initial_state_enceladus_fixed[0:3])
initial_state[3:6] = rotation_matrix.dot(initial_state_enceladus_fixed[3:6])

# Assign initial state in Keplerian elements in inertial frame, body-fixed frame, and cartesian body-fixed frame to check if conversions work correctly
initial_state_keplerian_inertial=np.ndarray([6])
initial_state_keplerian_fixed=np.ndarray([6])
initial_state_cartesian_fixed=np.ndarray([6])
initial_state_keplerian_inertial[0:6]=element_conversion.cartesian_to_keplerian(initial_state[0:6], enceladus_gravitational_parameter)
initial_state_keplerian_fixed[0:6]=element_conversion.cartesian_to_keplerian(initial_state_enceladus_fixed_list[0:6], enceladus_gravitational_parameter)
initial_state_cartesian_fixed[0:3]=rotation_matrix_back.dot(initial_state[0:3])
initial_state_cartesian_fixed[3:6]=rotation_matrix_back.dot(initial_state[3:6])

# Print initial state in different frames and coordinates to check if conversions work correctly
print("initial state keplerian inertial")
print(initial_state_keplerian_inertial)
print("initial state keplerian fixed")
print(initial_state_keplerian_fixed)
print("initial state cartesian inertial")
print(initial_state)
print("initial state cartesian fixed")
print(initial_state_cartesian_fixed)

### Define dependent variables to save

# Define list of dependent variables to save
dependent_variables_to_save = ([
    propagation_setup.dependent_variable.total_acceleration("Orbiter"),
    propagation_setup.dependent_variable.keplerian_state("Orbiter", "Enceladus"),
    propagation_setup.dependent_variable.latitude("Orbiter", "Enceladus"),
    propagation_setup.dependent_variable.longitude("Orbiter", "Enceladus"),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.spherical_harmonic_gravity_type, "Orbiter", "Enceladus"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.spherical_harmonic_gravity_type, "Orbiter", "Saturn"
    )
])

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

# Plot ground track for the simulation period
latitude = dep_vars_array[:,10]
longitude = dep_vars_array[:,11]
hours = (simulation_end_epoch-simulation_start_epoch)/3600 #31
subset = int(len(time_hours) / 24 * hours)
latitude = np.rad2deg(latitude[0: subset])
longitude = np.rad2deg(longitude[0: subset])
plt.figure(figsize=(9, 5))
plt.title("Ground track of Orbiter")
#plt.plot(longitude,latitude, linewidth=0.5)
plt.scatter(longitude, latitude, marker='.', s=0.05)
plt.xlabel('Longitude [deg]')
plt.ylabel('Latitude [deg]')
plt.xlim([min(longitude), max(longitude)])
plt.yticks(np.arange(-90, 91, step=45))
plt.grid()
plt.tight_layout()
plt.show()

### Conversion of Cartesian elements to the body-fixed frame, then conversion to Kepler elements

num_rows = np.shape(states_array)[0]
num_columns = np.shape(states_array)[1]

states_array_body_fixed_frame = np.ndarray([num_rows,7])

for i in range(num_rows):
    states_array_body_fixed_frame[i,0] = states_array[i,0]
    states_array_body_fixed_frame[i,1:4] = rotation_matrix_back.dot(states_array[i,1:4])
    states_array_body_fixed_frame[i,4:7] = rotation_matrix_back.dot(states_array[i,4:7])

kepler_body_fixed = np.ndarray([num_rows,7])

kepler_body_fixed[i,0] = states_array_body_fixed_frame[i,0]

for i in range(num_rows):
     kepler_body_fixed[i,1:7] = element_conversion.cartesian_to_keplerian(states_array_body_fixed_frame[i,1:7], enceladus_gravitational_parameter)

### Kepler elements over time
"""
Let's now plot each of the 6 Kepler element as a function of time, also as saved in the dependent variables.
"""

# Plot Kepler elements as a function of time
kepler_elements = dep_vars_array[:,4:10]
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(9, 12))
fig.suptitle('Evolution of Kepler elements over the course of the propagation.')

# Altitude
altitude = (kepler_body_fixed[:,1] - reference_radius_enc) / 1e3
ax1.plot(time_hours, altitude)
ax1.set_ylabel('Altitude [km]')

# Eccentricity
eccentricity = kepler_body_fixed[:,2]
ax2.plot(time_hours, eccentricity)
ax2.set_ylabel('Eccentricity [-]')

# Inclination
inclination = np.rad2deg(kepler_body_fixed[:,3])
ax3.plot(time_hours, inclination)
ax3.set_ylabel('Inclination [deg]')

# Argument of Periapsis
argument_of_periapsis = np.rad2deg(kepler_body_fixed[:,4])
ax4.plot(time_hours, argument_of_periapsis)
ax4.set_ylabel('Argument of Periapsis [deg]')

# Right Ascension of the Ascending Node
raan = np.rad2deg(kepler_body_fixed[:,5])
ax5.plot(time_hours, raan)
ax5.set_ylabel('RAAN [deg]')

# True Anomaly
true_anomaly = np.rad2deg(kepler_body_fixed[:,6])
ax6.scatter(time_hours, true_anomaly, s=1)
ax6.set_ylabel('True Anomaly [deg]')
ax6.set_yticks(np.arange(0, 361, step=60))

for ax in fig.get_axes():
    ax.set_xlabel('Time [hr]')
    ax.set_xlim([min(time_hours), max(time_hours)])
    ax.grid()
ax1.set_ylim([min((kepler_body_fixed[:,1] - reference_radius_enc) / 1e3)-5, max((kepler_body_fixed[:,1] - reference_radius_enc) / 1e3)+5])
plt.tight_layout()
plt.show()

### Accelerations over time

plt.figure(figsize=(9, 5))

# Spherical Harmonics Gravity Acceleration Enceladus
acceleration_norm_sh_enceladus = dep_vars_array[:,12]
plt.plot(time_hours, acceleration_norm_sh_enceladus, label='SH Enceladus')

# Point Mass Gravity Acceleration Saturn
acceleration_norm_sh_saturn = dep_vars_array[:,13]
plt.plot(time_hours, acceleration_norm_sh_saturn, label='SH Saturn')

plt.xlim([min(time_hours), max(time_hours)])
plt.xlabel('Time [hr]')
plt.ylabel('Acceleration Norm [m/s$^2$]')

plt.legend(bbox_to_anchor=(1.005, 1))
plt.suptitle("Accelerations norms on Orbiter, distinguished by type and origin, over the course of propagation.")
plt.yscale('log')
plt.grid()
plt.tight_layout()
plt.show()

print("states_array 7")
print(states_array[0])
print(states_array[num_rows-1])
print(states[simulation_end_epoch])

# Define a 3D figure using pyplot
fig = plt.figure(figsize=(6,6), dpi=125)
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f'Orbiter trajectory around Enceladus')

ax.set_proj_type('ortho')
ax.set_aspect('equal')

# Plot the positional state history
ax.scatter(0.0, 0.0, 0.0, label="Enceladus", marker='o', color='blue', s=100)
ax.plot(states_array_body_fixed_frame[0:2000, 1]/1000, states_array_body_fixed_frame[0:2000, 2]/1000, states_array_body_fixed_frame[0:2000, 3]/1000, label=bodies_to_propagate[0], linestyle='-', color='black', linewidth=0.3)
ax.view_init(0, 0)

# Add the legend and labels, then show the plot
ax.legend()
ax.set_xlabel('x [km]')
ax.set_ylabel('y [km]')
ax.set_zlabel('z [km]')
plt.show()
