# write_messages.py
from jinja2 import Environment, FileSystemLoader

input_width  = 22
output_width = 22

test_name = [
    {"name": "conv2_input_22x22", "kernel_size": 3},
    #{"name": "conv5", "kernel_size": 5},
    #{"name": "conv9", "kernel_size": 9},
]

loader      = FileSystemLoader('templates')
environment = Environment(loader = loader)
template    = environment.get_template("message.txt")


for test in test_name:
    filename = f"message_{test['name'].lower()}.txt"
    content = template.render(
        kernel_size = test['kernel_size'],
        width_size = input_width,
        height_size = output_width
    )
    with open(filename, mode="w", encoding="utf-8") as message:
        message.write(content)
        print(f"... wrote {filename}")