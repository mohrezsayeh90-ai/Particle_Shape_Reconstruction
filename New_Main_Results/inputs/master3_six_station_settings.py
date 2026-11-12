SIX_STATION_SETTINGS = dict(
    target_center=(0.86 * L, 0.50 * H),  # EDIT this (x,y) to move all three letters.
    width=0.15, height=0.19,             # Letter bounding-box sizes; never auto-shrunk.
    shapes=('S', 'C', 'T'),
    max_flight_age=8.0, min_flight_age=0.03, lead_time=0.25,
    seed_u_count=4, seed_v_count=15,     # Bank is sampled at the FIXED stations.
    fits_per_station_target=3, options_per_pair=3,
    pair_fit_nfev=32, matchings_to_test=48, matching_restarts=2,
    joint_starts=3, joint_nfev=45, direct_nfev=25,
    collision_samples=61, verify_samples=801, clearance_reserve=0.0003,
    position_tolerance=0.003, reference_tolerance=0.004, collision_margin=0.001,
    seed=512,
)
