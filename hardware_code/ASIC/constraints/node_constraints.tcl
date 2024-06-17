# ####################################################################

#  Created by Genus(TM) Synthesis Solution 17.11-s014_1 on Thu Nov 07 16:17:38 CET 2019

# ####################################################################

create_clock -name clk -period 440.0 [get_ports clk]
set_input_delay 0.01 -min -clock clk [remove_from_collection [all_inputs] [get_ports {clk}]] 
set_input_delay 1.00 -max -clock clk [remove_from_collection [all_inputs] [get_ports {clk}]] 

#set_false_path -from [get_ports reset] -to [all_registers]

set_output_delay 0.0 -clock clk [get_ports [all_outputs]]

set_driving_cell -lib_cell BUFFD12BWP -pin Z [remove_from_collection [all_inputs] {clk}]
set_load 0.01 [all_outputs]

