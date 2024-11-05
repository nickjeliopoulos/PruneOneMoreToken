These scripts are used to lock the maximum clock rate of the NVIDIA Jetson TX2 and NVIDIA Jetson AGX Orin (32GB) for measuring inference.

Specifically, these scripts set the devices to maximum power mode, which also fixes memory and CPU clock rates at their maximums.

On systems with administrator access, you can lock the clock rates of A100 with nvidia-smi. We were not able to do this with our machines, but we encourage others to do so if possible.

Examples: https://codeyarns.com/tech/2019-04-26-nvidia-smi-cheatsheet.html#gsc.tab=0