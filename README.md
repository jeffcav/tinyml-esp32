# tinyml-esp32

## Prepare environment

First, create a python environment and download required dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Show instructions:
~/dev/tinyml/tinyml-esp32/src/embedded/2-mlp-baremetal-int8/esp32s3$ ~/.espressif/tools/xtensa-esp32s3-elf/esp-2022r1-11.2.0/xtensa-esp32s3-elf/bin/xtensa-esp32s3-elf-objdump -d -S build/main.elf > instructions.txt