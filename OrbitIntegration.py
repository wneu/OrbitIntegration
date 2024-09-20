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
#matplotlib.use('Agg')  # Use a non-GUI backend
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

## Configuration

# Load spice kernels mentioned in Benedikter et al. (2022)
#spice.load_standard_kernels()
spice.load_kernel('C:\TudatProjects\OrbitIntegration\de438.bsp') # https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de438.bsp 2019-08-28 10:42 114M
spice.load_kernel('C:\TudatProjects\OrbitIntegration\sat427.bsp') #https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/a_old_versions/sat427.bsp 2020-01-27 12:22 242M
spice.load_kernel('C:\TudatProjects\OrbitIntegration\pck00010.tpc') #https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00010.tpc 2011-10-21 17:41 123K
spice.get_total_count_of_kernels_loaded()

# Set simulation start and end epochs; this is one period of the K2 orbit from Benedikter et al. (2022)
simulation_start_epoch = DateTime(2000, 1, 1).epoch()
simulation_end_epoch   = DateTime(2000, 1, 4, 1, 55, 6).epoch()

#print(simulation_start_epoch)
#print(simulation_end_epoch)

## Environment setup
"""
Create the environment for the simulation. This covers the creation of (celestial) bodies, vehicle(s), and environment interfaces.
"""

### Create the bodies

# Define string names for bodies to be created from default.
bodies_to_create = ["Enceladus", "Saturn"]

# Use "ECLIPJ2000" as global frame origin and orientation.
global_frame_origin = "Enceladus"
global_frame_orientation = "J2000"

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
    [ #Iess et al. 2014, but as in the minimal example by Andreas with c20=J2 and c30=J3 (wrong signs)
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [-5.4352E-03, 9.2E-06, 1.5498E-03, 0],
        [1.15E-04, 0, 0, 0]
    ],
    [ #Iess et al. 2014, as in the minimal example by Andreas
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
saturn_gravitational_parameter = 3.7931208E+16
saturn_reference_radius = 60330000.0

# Normalize the spherical harmonic coefficients
nor_sh_sat=astro.gravitation.normalize_spherical_harmonic_coefficients(
    [ #Iess et al. 2019, as in the minimal example by Andreas
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
    [ #Iess et al. 2019, as in the minimal example by Andreas
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

# Create ground stations
# Define the positions of the ground stations on Enceladus
s1_altitude = 0.0
s2_altitude = 0.0
s3_altitude = 0.0
s1_latitude = np.deg2rad(0)
s2_latitude = np.deg2rad(0)
s3_latitude = np.deg2rad(10)
s1_longitude = np.deg2rad(0)
s2_longitude = np.deg2rad(10)
s3_longitude = np.deg2rad(5)

# Create ground station settings
ground_station_settings = environment_setup.ground_station.basic_station(
    "TrackingStation1",
     [s1_altitude, s1_latitude, s1_longitude],
     element_conversion.geodetic_position_type)
ground_station_settings = environment_setup.ground_station.basic_station(
    "TrackingStation2",
     [s2_altitude, s2_latitude, s2_longitude],
     element_conversion.geodetic_position_type)
ground_station_settings = environment_setup.ground_station.basic_station(
    "TrackingStation3",
     [s3_altitude, s3_latitude, s3_longitude],
     element_conversion.geodetic_position_type)

# Append station settings to existing (default is empty) list
body_settings.get( "Enceladus" ).ground_station_settings.append( ground_station_settings )

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

# Define accelerations acting on Orbiter by Saturn, and Enceladus, as in the minimal example by Andreas
accelerations_settings_orbiter = dict(
    Enceladus=[
        #propagation_setup.acceleration.point_mass_gravity()
        propagation_setup.acceleration.spherical_harmonic_gravity(3, 3)
    ],
    Saturn=[
        propagation_setup.acceleration.spherical_harmonic_gravity(8, 8)
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

# Orbit K2 from Benedikter et al. (2022), minimal example
# initial elements [a [km], e, i [deg], Omega [deg], omega [deg], Ma [deg]] (the order is Omega, omega, not omega, Omega)
# sv_start = np.array([4.90400000e+02, 7.59474232e-02, 5.62500000e+01, 2.41719177e+02, 2.68144032e+02, 8.31499110e+01])
# 475.323709  102.991720  -48.576955  0.003009  0.075062  0.095705
"""initial_state_enceladus_fixed_list = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=enceladus_gravitational_parameter,
    semi_major_axis=0.490400e+06,
    eccentricity=0.759474232e-01,
    inclination=np.deg2rad(5.62500000e+01),
    argument_of_periapsis=np.deg2rad(2.68144032e+02),
    longitude_of_ascending_node=np.deg2rad(2.41719177e+02),
    true_anomaly=np.deg2rad(8.31499110e+01),
)"""

# Assign initial state in Cartesian coordinates in body-fixed frame
#initial_state_enceladus_fixed = np.array(initial_state_enceladus_fixed_list)

# Get rotation matrices between IAU_Enceladus and global_frame_orientation
rotation_matrix = spice.compute_rotation_matrix_between_frames("IAU_Enceladus",global_frame_orientation, simulation_start_epoch )
rotation_matrix_back = spice.compute_rotation_matrix_between_frames(global_frame_orientation,"IAU_Enceladus", simulation_start_epoch )

# Assign initial state in Cartesian coordinates in inertial frame
initial_state = np.ndarray([6])
initial_state[0:3] = [475323.709, 102991.720, -48576.955] #rotation_matrix.dot(initial_state_enceladus_fixed[0:3])
initial_state[3:6] = [3.009, 75.062, 95.705] #rotation_matrix.dot(initial_state_enceladus_fixed[3:6])
#initial_state[0:3] = rotation_matrix.dot(initial_state_enceladus_fixed[0:3])
#initial_state[3:6] = rotation_matrix.dot(initial_state_enceladus_fixed[3:6])

# Assign initial state in Keplerian elements in inertial frame, body-fixed frame, and cartesian body-fixed frame to check if conversions work correctly
initial_state_keplerian_inertial=np.ndarray([6])
initial_state_keplerian_fixed=np.ndarray([6])
initial_state_cartesian_fixed=np.ndarray([6])
initial_state_keplerian_inertial[0:6]=element_conversion.cartesian_to_keplerian(initial_state[0:6], enceladus_gravitational_parameter)
#initial_state_keplerian_fixed[0:6]=element_conversion.cartesian_to_keplerian(initial_state_enceladus_fixed_list[0:6], enceladus_gravitational_parameter)
initial_state_cartesian_fixed[0:3]=rotation_matrix_back.dot(initial_state[0:3])
initial_state_cartesian_fixed[3:6]=rotation_matrix_back.dot(initial_state[3:6])
initial_state_keplerian_fixed[0:6]=element_conversion.cartesian_to_keplerian(initial_state_cartesian_fixed[0:6], enceladus_gravitational_parameter)

# Print initial state in different frames and coordinates to check if conversions work correctly for the initial state
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
    ),
    propagation_setup.dependent_variable.altitude("Orbiter", "Enceladus")
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

### Total acceleration over time

# Plot total acceleration as function of time
time_hours = dep_vars_array[:,0]/3600
total_acceleration_norm = np.linalg.norm(dep_vars_array[:,1:4], axis=1)
plt.figure(figsize=(9, 5), dpi=400)
plt.title("Total acceleration norm on Orbiter over the course of propagation.")
plt.plot(time_hours, total_acceleration_norm, color='black', linewidth=1)
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
hours = (simulation_end_epoch-simulation_start_epoch)/3600 # not used currently
subset = int(len(time_hours) / 24 * hours) # not used currently
latitude = np.rad2deg(latitude)
longitude = np.rad2deg(longitude)
plt.figure(figsize=(9, 5), dpi=400)
plt.title("Ground track of Orbiter")
#plt.plot(longitude, latitude, linewidth=0.5)
plt.scatter(longitude, latitude, marker='.', s=0.05, c='black')
plt.scatter(np.rad2deg(s1_longitude), np.rad2deg(s1_latitude))
plt.scatter(np.rad2deg(s2_longitude), np.rad2deg(s2_latitude))
plt.scatter(np.rad2deg(s3_longitude), np.rad2deg(s3_latitude))
plt.xlabel('Longitude [deg]')
plt.ylabel('Latitude [deg]')
plt.xticks(np.arange(-150, 200, step=50))
plt.xlim([min(longitude), max(longitude)])
plt.yticks(np.arange(-80, 100, step=20))
plt.ylim([-90, 90])
plt.grid()
plt.tight_layout()
plt.show()

### Conversion of Cartesian elements to the body-fixed frame for each time step, then conversion to Kepler elements

num_rows = np.shape(states_array)[0]
num_columns = np.shape(states_array)[1]

states_array_body_fixed_frame = np.ndarray([num_rows,7])

for i in range(num_rows):
    rotation_matrix_back = spice.compute_rotation_matrix_between_frames(global_frame_orientation, "IAU_Enceladus", states_array[i,0])
    states_array_body_fixed_frame[i,0] = states_array[i,0]
    states_array_body_fixed_frame[i,1:4] = rotation_matrix_back.dot(states_array[i,1:4])
    states_array_body_fixed_frame[i,4:7] = rotation_matrix_back.dot(states_array[i,4:7])

kepler_body_fixed = np.ndarray([num_rows,7])

kepler_body_fixed[i,0] = states_array_body_fixed_frame[i,0]

for i in range(num_rows):
     kepler_body_fixed[i,1:7] = element_conversion.cartesian_to_keplerian(states_array_body_fixed_frame[i,1:7], enceladus_gravitational_parameter)

### Kepler elements over time
"""
Let's now plot the altitude and each of the 6 Kepler element as a function of time, also as saved in the dependent variables.
"""

# Plot Kepler elements as a function of time
kepler_elements = dep_vars_array[:,4:10]
fig, ((ax1), (ax2), (ax3), (ax4), (ax5), (ax6), (ax7)) = plt.subplots(7, 1, figsize=(12, 12))
fig.suptitle('Evolution of Kepler elements over the course of the propagation.')

# Altitude
altitude = dep_vars_array[:,14]/1E3
ax1.plot(time_hours, altitude, color='black')
ax1.set_ylabel('Altitude [km]')

# Semi-Major Axis
#semi_major_axis = dep_vars_array[:,4]/1e3#
#semi_major_axis = (kepler_elements[:,0]) / 1e3
semi_major_axis = kepler_body_fixed[:,1] / 1e3
ax2.plot(time_hours, semi_major_axis, color='black')
ax2.set_ylabel('Semi-Major Axis [km]')

# Eccentricity
eccentricity = dep_vars_array[:,5]
#eccentricity = kepler_body_fixed[:,2]
ax3.plot(time_hours, eccentricity, color='black')
ax3.set_ylabel('Eccentricity [-]')

# Inclination
inclination = np.rad2deg(kepler_elements[:,2])
#inclination = np.rad2deg(kepler_body_fixed[:,3])
ax4.plot(time_hours, inclination, color='black')
ax4.set_ylabel('Inclination [deg]')

# Argument of Periapsis
argument_of_periapsis = np.rad2deg(kepler_body_fixed[:,4])
ax5.plot(time_hours, argument_of_periapsis, color='black')
ax5.set_ylabel('Argument of Periapsis [deg]')

# Right Ascension of the Ascending Node
raan = np.rad2deg(kepler_body_fixed[:,5])
ax6.plot(time_hours, raan, color='black')
ax6.set_ylabel('RAAN [deg]')

# True Anomaly
true_anomaly = np.rad2deg(kepler_body_fixed[:,6])
ax7.scatter(time_hours, true_anomaly, s=1, c='black')
ax7.set_ylabel('True Anomaly [deg]')
ax7.set_yticks(np.arange(0, 361, step=60))

for ax in fig.get_axes():
    ax.set_xlabel('Time [hr]')
    ax.set_xlim([min(time_hours), max(time_hours)])
    ax.grid()
ax1.set_ylim([min(altitude)-5, max(altitude)+5])
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

# Define a 3D figure using pyplot
fig = plt.figure(figsize=(6,6), dpi=400)
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f'Orbiter trajectory around Enceladus')
ax.set_proj_type('ortho')
ax.set_aspect('equal')
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Conversion of the ground station positions to Cartesian coordinates
pos_gs=np.ndarray([3,3])
pos_gs[0,:]=[((s1_altitude+reference_radius_enc)/1e3)*math.sin(90.-s1_latitude)*math.cos(s1_longitude), ((s1_altitude+reference_radius_enc)/1e3)*math.sin(90.-s1_latitude)*math.sin(s1_longitude), ((s1_altitude+reference_radius_enc)/1e3)*math.cos(90.-s1_latitude)]
pos_gs[1,:]=[((s2_altitude+reference_radius_enc)/1e3)*math.sin(90.-s2_latitude)*math.cos(s2_longitude), ((s2_altitude+reference_radius_enc)/1e3)*math.sin(90.-s2_latitude)*math.sin(s2_longitude), ((s2_altitude+reference_radius_enc)/1e3)*math.cos(90.-s2_latitude)]
pos_gs[2,:]=[((s3_altitude+reference_radius_enc)/1e3)*math.sin(90.-s3_latitude)*math.cos(s3_longitude), ((s3_altitude+reference_radius_enc)/1e3)*math.sin(90.-s3_latitude)*math.sin(s3_longitude), ((s3_altitude+reference_radius_enc)/1e3)*math.cos(90.-s3_latitude)]

# Plot the positional state history including ground stations
ax.scatter(0.0, 0.0, 0.0, marker='o', color='blue', s=100)
ax.plot(states_array_body_fixed_frame[:, 1]/1000, states_array_body_fixed_frame[:, 2]/1000, states_array_body_fixed_frame[:, 3]/1000, linestyle='-', color='black', linewidth=0.5)
#ax.scatter(pos_gs[:,0], pos_gs[:,1], pos_gs[:,2], marker='o', color='red', s=10)
# Add the legend and labels, then show the plot
#ax.legend()
#ax.set_xlabel('x [km]')
ax.set_xlim(-500, 500)
ax.set_ylim(-500, 500)
ax.set_zlim(-500, 500)
ax.set_ylabel('y [km]')
ax.set_zlabel('z [km]')
ax.set_xticks([])
ax.set_yticks([-400, -200, 0, 200, 400])
ax.set_zticks([-400, -200, 0, 200, 400])
#ax.grid(False)
ax.view_init(0, 0)
plt.show()




#print(
#    f"""
#Single Enceladus-Orbiting Satellite Example.
#The initial time in seconds is: \n{
#    states_array_body_fixed_frame[0,0]} \n
#The initial position vector of Orbiter is [km]: \n{
#    states_array[0,1:4] / 1E3} \n{
#    states_array_body_fixed_frame[0, 1:4] / 1E3} \n
#The initial velocity vector of Orbiter is [km/s]: \n{
#    states_array[0,4:7] / 1E3} \n
#After {simulation_end_epoch} seconds the final time in seconds is: \n{
#    states_array_body_fixed_frame[num_rows-1,0]} \n{
#    simulation_end_epoch-simulation_start_epoch
#    }\n
#And the position vector of Orbiter is [km]: \n{
#    states_array[num_rows-1,1:4] / 1E3} \n{
#    states_array_body_fixed_frame[num_rows-1,1:4] / 1E3} \n
#And the velocity vector of Orbiter is [km/s]: \n{
#    states_array[num_rows-1,4:7] / 1E3} \n
#    """
#)

"""from mpl_toolkits.mplot3d import axes3d
fig = plt.figure('orbit', figsize=(6,6))
plt.clf()
ax = plt.axes(projection='3d')
ax.set_proj_type('ortho')
ax.set_aspect('equal')
# Make Ball
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
r_e = 256300*1e-3
x = r_e * np.outer(np.cos(u), np.sin(v))
y = r_e * np.outer(np.sin(u), np.sin(v))
z = r_e * np.outer(np.ones(np.size(u)), np.cos(v))
#ax.plot_surface(x, y, z, color='lightskyblue')
#ax.plot3D(states_array_body_fixed_frame[:, 1]/1000, states_array_body_fixed_frame[:, 2]/1000, states_array_body_fixed_frame[:, 3]/1000, color='black', linewidth=0.5, zorder=10)
ax.plot(states_array_body_fixed_frame[:, 1]/1000, states_array_body_fixed_frame[:, 2]/1000, states_array_body_fixed_frame[:, 3]/1000, linestyle='-', color='black', linewidth=0.5)
#ax.scatter(pos_gs[:,0], pos_gs[:,1], pos_gs[:,2], marker='o', color='red', s=10)
ax.set_xlim(-500, 500)
ax.set_ylim(-500, 500)
ax.set_zlim(-500, 500)
ax.set_ylabel('y [km]')
ax.set_zlabel('z [km]')
ax.set_xticks([])
ax.set_yticks([-400, -200, 0, 200, 400])
ax.set_zticks([-400, -200, 0, 200, 400])
#ax.grid(False)
ax.view_init(0, 0)
plt.locator_params(nbins=5)
# plt.savefig('Plots/Orbit/orbit3d2.png', bbox_inches='tight', dpi=600)
plt.show()
"""
