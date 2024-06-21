<h1 align="center">RL-based MIG Scheduler</h1>
<p align="center"><i>GPU is all you need. but efficient.</i></p>
<div align="center">

![Static Badge](https://img.shields.io/badge/python-3.10-blue)

</div>

RL-based MIG Scheduler(RMS) is a development project of basic framework and API implementation it designed to efficient NVIDIA Multi-Instance GPU(MIG) scheduling with reinforcement learning. Intelligence of RMS utilize maximally exist GPU device with MIG profile reconfiguration considering future resource demand to avoid as possible as provisioning new GPU device it caused by no slot is there which satisfy resource requirement.

## Getting Started
### Preconditions
The GI scheduling and MIG configuration changes based on the resource size of the requests considered by RMS, as well as the lifecycle GI inventory management, are premised on the satisfaction of the following conditions:
- Using an NVIDIA A100 80G PCIe GPU
- Policy prevents users from using more than one GI at the same time

### Installation
```
git clone https://github.com/Yuri-0/RL-based-GPU-scheduler-for-Efficient-NVIDIA-MIG.git
pip install -r requirements.txt
```

### Usage
Start with python execution of `gpu_scheduler_api.py`. Input would be take `STDIN` and output `STDOUT`.
```
python gpu_scheduler_api.py

# GPU allocation request
acquire <Username> <GPU Spec>

# GPU de-Username request
release <Username>
```
To allocate specific GPU resource for who is username, use `acquire` command. And to deallocate GPU resource for who is username, use `release` command. Each arguments for the command denoted as followed.
- **Username**: Any string that does not contain spaces.
- **GPU Spec**: The number of GPC slices, one of `1g`, `2g`, `3g`, `4g`, or `7g`.

By the preconditions, if which user get GPU resource by `acquire` command, cannot use the command again until deallocate the resource with `release` command.

## How it work
The agent of RMS, which bind GPU Instance(GI) request if slot exist satisfying resource requirement, else if reconfigurable GPU are there set a new MIG profile after prediction. It trained in scenario that random GI request come, however you can use your own dataset about time-series GI request.


In most cases, the new MIG profile predicted by the RMS agent can be bound the GI requests. However, due to the imperfections of the RMS agent, if the predicted MIG profile does not immediately provide a slot that meets the resource requirements, it will instead be set to the MIG profile that is most similar to the predicted one and includes a slot that meets the resource requirements. 

### Example
In the case of allocating `3g` of GPU resources to `user-1`, the input and output would be the same as follows, based on the description in the Usage section:
```
$ acquire user-1 3g
0 5
```
Output is the allocated GPU information in the form of `<GPU Id> <MIG Config Id>`.
- **GPU Id**: The ID of the GPU to process the request. An integer starting at 0.
- **MIG Config Id**: Config number as it appears in the MIG scheme.

RMS checks whether the requested resource can be provided from the GI inventory. Since the `3g` size resource cannot be provided from the existing GI inventory, it provisions a new GPU, `GPU Id 0`.

The RMS agent decides on `MIG Config Id 5`, which includes a `3g` size GI and is expected to meet future requests without additional GPU provisioning, utilizing the inventory secured through this provisioning.

Finally RMS sets the MIG config of the new GPU to `MIG Config Id 5` and allocates a `3g` size GI from the created GI inventory to user-1.

## TODO
- [ ] Improve interface as API from STDIN/STDOUT
- [ ] Improve reward policy in training of RL agent
- [ ] Consider multiple resource requests from a single user
- [ ] Respond to additional MIG-enabled GPUs (H100 and others)
- [ ] Write tutorials on applying infrastructure systems
- [ ] Implement online learning features구현

## Authors
- **Gijun Park** (*Machine Learning Research Scientist @Okestro*) - [LinkedIn](www.linkedin.com/in/gijun-park-ba4281249)
