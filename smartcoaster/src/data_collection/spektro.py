import AS7262_Pi_bus4 as spec4
import time

#as7262 = spec4()
spec4.soft_reset()
spec4.set_gain(3)
spec4.set_integration_time(255)
spec4.set_measurement_mode(2)
#as7262.set_illumnination_led_current(12.5)
spec4.set_led_current(3)
spec4.enable_main_led()

def main():
    try:
        while 1:
            values = spec4.get_calibrated_values()
            time.sleep(1)
            print(spec4.get_calibrated_values())
            print("")
            #spec = [float(1) for i in list(values)]
            #print(spec)
    except KeyboardInterrupt:
        spec4.set_measurement_mode(3)
        spec4.disable_main_led()
        #spec4.set_illumination_mode(3)
        #spec4.set_illumination_led(0)

if __name__ == '__main__':
    main()
