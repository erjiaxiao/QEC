create_clock -period 440.000 -name clk -waveform {0.000 220.000} [get_ports clk]
set_input_delay -clock [get_clocks clk] -min -add_delay 0.000 [get_ports {inputs_t[*]}]
set_input_delay -clock [get_clocks clk] -max -add_delay 2.000 [get_ports {inputs_t[*]}]
set_input_delay -clock [get_clocks clk] -min -add_delay 0.000 [get_ports reset]
set_input_delay -clock [get_clocks clk] -max -add_delay 2.000 [get_ports reset]
set_output_delay -clock [get_clocks clk] -min -add_delay -1.000 [get_ports {outputs_t[*]}]
set_output_delay -clock [get_clocks clk] -max -add_delay 2.000 [get_ports {outputs_t[*]}]
