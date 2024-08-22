import tkinter
from tkinter import ttk
from tkinter import filedialog
import tkinter.messagebox
from utils import *

user_config = User_Input()

def openFile():
    filepath = filedialog.askopenfilename(initialdir="C:\\Users\\45526\Desktop\\Master\\Second Year\\Thesis\\Project\\processor_simulator",
                                          title="Select a NN model",
                                          filetypes=(("ONNX files", "*.onnx"), ("All files", "*.*")))
    nn_model_var.set(filepath)

def enter_data():
    # Flag to check if the input was correctly defined
    error_status = 0

    # Get the user input
    pipeline = pipeline_enable_combobox.get()
    pipeline_stages = pipeline_stages_entry.get()
    data_forwarding = data_forwarding_combobox.get()
    loop_unrolling = loop_unrollling_combobox.get()
    loop_unrolling_factor = loop_unrollling_factor.get()
    addition_cc = addition.get()
    multiplication_cc = multiplication.get()
    division_cc = division.get()
    SIMD_status = SISD_SIMD_combobox.get()
    VLIW_status = VLIW_combobox.get()
    nr_vector_units = number_of_vector_units.get()
    nr_vector_lanes = number_of_vector_lanes.get()

    # TODO: Create a function for checking mechanism
    # Pipeline checking
    if pipeline == "":
        tkinter.messagebox.showerror(title = "Pipeline error", message = "Please select an option for the pipeline!")
    elif pipeline == "Disabled" and pipeline_stages != "0":
        tkinter.messagebox.showerror(title = "Pipeline error", message = "The pipeline is disabled,but the number of pipeline stages is different than 0!")
        error_status = 1
    elif pipeline == "Enabled" and pipeline_stages == "0":
        tkinter.messagebox.showerror(title = "Pipeline error", message = "The pipeline is enabled, but the number of pipeline stages is 0!")
        error_status = 1
        
    # Loop unrolling checking
    elif loop_unrolling == "":
        tkinter.messagebox.showerror(title = "Loop unrolling error", message = "Please select an option for the loop unrolling!")
    elif loop_unrolling == "Disabled" and loop_unrolling_factor != "0":
        tkinter.messagebox.showerror(title = "Loop unrolling error", message = "The loop unrolling mechanism is disabled, but the factor is different than 0!")
        error_status = 1
    elif loop_unrolling == "Enabled" and loop_unrolling_factor == "0":
        tkinter.messagebox.showerror(title = "Loop unrolling error", message = "The loop unrolling mechanism is enabled, but the factor is 0!")
        error_status = 1

    # Data forwarding checking
    elif data_forwarding == "":
        tkinter.messagebox.showerror(title = "Data forwarding error", message = "Please select an option for the data forwarding!")

    # Hardware accelerators checking
    elif SIMD_status == "":
        tkinter.messagebox.showerror(title = "SIMD error", message = "Please select an option for the SIMD!")
    elif VLIW_status == "":
        tkinter.messagebox.showerror(title = "VLIW error", message = "Please select an option for the VLIW!")
    elif SIMD_status == "Enabled" and VLIW_status == "Enabled":
        tkinter.messagebox.showerror(title = "Hardware accelerators error", message = "Both hardware accelerators are enabled!")
        error_status = 1

    else:
        if error_status == 0:
            print("*********************************************")
            print("The processor was configured:")
            print("Pipeline: ", pipeline, ", Number of pipeline stages: ", pipeline_stages)
            print("Data forwarding: ", data_forwarding)
            print("Loop unrolling: ", loop_unrolling, ", Factor: ", loop_unrolling_factor)
            print("Addition: ", addition_cc)
            print("Multiplicaiton: ", multiplication_cc)
            print("Division: ", division_cc)
            print("SIMD status: ", SIMD_status)
            print("VLIW status: ", VLIW_status)
            print("Number of vector_units: ", nr_vector_units)
            print("Number of vector_lanes: ", nr_vector_lanes)
            print("*********************************************")

            # Cast to integer
            pipeline_stages   = int(pipeline_stages)
            loop_factor       = int(loop_unrolling_factor)
            addition_cc       = int(addition_cc)
            multiplication_cc = int(multiplication_cc)
            division_cc       = int(division_cc)
            nr_vector_units         = int(nr_vector_units)
            nr_vector_lanes    = int(nr_vector_lanes)

            # Create the user_input object
            user_config.set_parameters(pipeline, pipeline_stages, loop_unrolling, loop_factor, data_forwarding, addition_cc, multiplication_cc,
                                       division_cc, SIMD_status, VLIW_status,  nr_vector_units, nr_vector_lanes)
            
            print(user_config)

# TODO: Create a Class for the GUI
# Construct the GUI
window = tkinter.Tk()
window.title("Processor Simulator for Neural Architecture Search")
frame = tkinter.Frame(window, background="#F8E5AB")
frame.pack()

# General information
general_info_frame = tkinter.LabelFrame(frame, text = "General information", background="#FEF3D9", labelanchor="n")
general_info_frame.grid(row = 0, column = 0, sticky ="news", padx = 20, pady = 10)

# Pipeline activation
pipeline_enable_label = tkinter.Label(general_info_frame, text = "Pipeline", background="#FEF3D9")
pipeline_enable_label.grid(row = 0, column = 0)
pipeline_enable_combobox = ttk.Combobox(general_info_frame, values = ["Enabled", "Disabled"], state = "readonly")
pipeline_enable_combobox.grid(row = 1 , column = 0,  padx = 60, pady = 5)

# Pipeline stages
pipeline_stages_label = tkinter.Label(general_info_frame, text = "Number of pipeline stages", background="#FEF3D9")
pipeline_stages_label.grid(row = 0, column = 1)
# Default value
pipeline_stages_var = tkinter.StringVar(value = "0")
pipeline_stages_entry = tkinter.Entry(general_info_frame, textvariable = pipeline_stages_var)
pipeline_stages_entry.grid(row = 1, column = 1)


# Features
features_frame = tkinter.LabelFrame(frame, text = "Processor features", background="#FEF3D9", labelanchor="n")
features_frame.grid(row = 1, column = 0, sticky ="news", padx = 20, pady = 10)

# Data forwarding
data_forwarding_label = tkinter.Label(features_frame, text = "Data forwarding", background="#FEF3D9")
data_forwarding_label.grid(row = 0, column = 0)
data_forwarding_combobox = ttk.Combobox(features_frame, values = ["", "Enabled", "Disabled"], state = "readonly")
data_forwarding_combobox.grid(row = 1 , column = 0, padx=50)


# Loop unrolling
loop_unrolling_label = tkinter.Label(features_frame, text = "Loop unrolling", background="#FEF3D9")
loop_unrolling_label.grid(row = 0, column = 1)
loop_unrollling_combobox = ttk.Combobox(features_frame, values = ["", "Enabled", "Disabled"], state = "readonly")
loop_unrollling_combobox.grid(row = 1 , column = 1)

# Loop unrolling factor
loop_unrolling_factor_label = tkinter.Label(features_frame, text = "Loop unrolling factor", background="#FEF3D9")
loop_unrolling_factor_label.grid(row = 0, column = 2)
# Default value
loop_factor_var = tkinter.StringVar(value = "0")
# Entry widget
loop_unrollling_factor = tkinter.Entry(features_frame, textvariable = loop_factor_var)
loop_unrollling_factor.grid(row = 1 , column = 2)

for widget in features_frame.winfo_children():
    widget.grid_configure(padx = 10, pady = 5)


# Timings
timings_frame = tkinter.LabelFrame(frame, text = "Timings configurations (CC)", background="#FEF3D9", labelanchor="n")
timings_frame.grid(row = 2, column = 0, sticky ="news", padx = 20, pady = 10)

# Addition
addition_label = tkinter.Label(timings_frame, text = "Addition", background="#FEF3D9")
addition_label.grid(row = 0, column = 0)
# Default value
addition_var = tkinter.StringVar(value = "0")
# Entry widget
addition = tkinter.Entry(timings_frame, textvariable = addition_var)
addition.grid(row = 1 , column = 0)


# Multiplication
multiplication_label = tkinter.Label(timings_frame, text = "Multiplication", background="#FEF3D9")
multiplication_label.grid(row = 0, column = 1)
# Default value
multiplication_var = tkinter.StringVar(value = "0")
# Entry widget
multiplication = tkinter.Entry(timings_frame, textvariable = multiplication_var)
multiplication.grid(row = 1 , column = 1)


# Division
division_label = tkinter.Label(timings_frame, text = "Division", background="#FEF3D9")
division_label.grid(row = 0, column = 2)
# Default value
division_var = tkinter.StringVar(value = "0")
# Entry widget
division = tkinter.Entry(timings_frame, textvariable = division_var)
division.grid(row = 1 , column = 2)

for widget in timings_frame.winfo_children():
    widget.grid_configure(padx = 10, pady = 5)


# Hardware accelerators parametrization
accelerators_frame = tkinter.LabelFrame(frame, text = "Hardware accelerators parametrization", background="#FEF3D9", labelanchor="n")
accelerators_frame.grid(row = 3, column = 0, sticky ="news", padx = 20, pady = 10)

# Activate the SIMD features
SISD_SIMD_label = tkinter.Label(accelerators_frame, text = "SIMD", background="#FEF3D9")
SISD_SIMD_label.grid(row = 0, column = 0)
SISD_SIMD_combobox = ttk.Combobox(accelerators_frame, values = ["Enabled", "Disabled"], state = "readonly")
SISD_SIMD_combobox.grid(row = 1 , column = 0, padx = 60, pady = 5)

# Activate the VLIW features
VLIW_label = tkinter.Label(accelerators_frame, text = "VLIW", background="#FEF3D9")
VLIW_label.grid(row = 0, column = 1)
VLIW_combobox = ttk.Combobox(accelerators_frame, values = ["Enabled", "Disabled"], state = "readonly")
VLIW_combobox.grid(row = 1 , column = 1)

# Configure the number of vector_units
number_of_vector_units_label = tkinter.Label(accelerators_frame, text = "Number of vector units", background="#FEF3D9")
number_of_vector_units_label.grid(row = 2, column = 0)
# Default value
vector_units_var = tkinter.StringVar(value = "0")
# Entry widget
number_of_vector_units = tkinter.Entry(accelerators_frame, textvariable = vector_units_var)
number_of_vector_units.grid(row = 3 , column = 0)

# Configure the number of vector_lanes
number_of_vector_lanes_label = tkinter.Label(accelerators_frame, text = "Number of vector lanes", background="#FEF3D9")
number_of_vector_lanes_label.grid(row = 2, column = 1)
# Default value
vector_lanes_var = tkinter.StringVar(value = "0")
# Entry widget
number_of_vector_lanes = tkinter.Entry(accelerators_frame, textvariable = vector_lanes_var)
number_of_vector_lanes.grid(row = 3 , column = 1)

# Configure the vector_width
vrf_width_label = tkinter.Label(accelerators_frame, text = "VRF width", background="#FEF3D9")
vrf_width_label.grid(row = 4, column = 0)
# Default value
vrf_width_var = tkinter.StringVar(value = "0")
# Entry widget
vrf_width = tkinter.Entry(accelerators_frame, textvariable = vrf_width_var)
vrf_width.grid(row = 5 , column = 0)


# Configure the execution_width
alu_width_label = tkinter.Label(accelerators_frame, text = "ALU width", background="#FEF3D9")
alu_width_label.grid(row = 4, column = 1)
# Default value
alu_width_var = tkinter.StringVar(value = "0")
# Entry widget
alu_width = tkinter.Entry(accelerators_frame, textvariable = alu_width_var)
alu_width.grid(row = 5 , column = 1)



# ONNX model
model_frame = tkinter.LabelFrame(frame, text = "Neural network models", background="#FEF3D9", labelanchor="n")
model_frame.grid(row = 4, column = 0, sticky ="news", padx = 20, pady = 10)


nn_model = tkinter.Label(model_frame, text = "Model selected:", background="#FEF3D9")
nn_model.grid(row = 0, column = 0)
nn_model_var = tkinter.StringVar()
nn_model_var_entry = tkinter.Entry(model_frame, textvariable = nn_model_var,width=55)
nn_model_var_entry.grid(row = 0, column = 1)

# Button
button = tkinter.Button(model_frame, text = "Open", command = openFile, background="#FFCE88")
button.grid(row = 1, column = 0, columnspan=2, sticky = "news")

for widget in model_frame.winfo_children():
    widget.grid_configure(padx = 10, pady = 5)

# Button
button = tkinter.Button(frame, text = "Enter data", command = enter_data, background="#FFCE88", activeforeground="#FEF3D9")
button.grid(row = 5, column = 0, sticky = "news", padx = 35, pady = 10)

window.mainloop()