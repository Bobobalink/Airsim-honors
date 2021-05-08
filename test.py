from BaseRacer import BaseRacer


if __name__ == '__main__':
    racer = BaseRacer()
    # racer.load_level('ZhangJiaJie_Medium')
    racer.load_level('Soccer_Field_Medium')
    racer.start_race(1)
    racer.initialize_drone()
    racer.takeoff_with_moveOnSpline()
    racer.get_ground_truth_gate_poses()
    racer.start_odometry_callback_thread()
    racer.fly_through_all_gates_at_once_with_moveOnSplineVelConstraints().join()
