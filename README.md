# Novel Caloric Restriction Mimetics (CRM) Generation Pipeline

Calorie restriction (CR) has been recognized as a beneficial practice for promoting healthy aging by preserving cellular function and extending the health span. To mimic the effects of caloric restriction, researchers have explored the use of Caloric Restriction Mimetics (CRMs), which are compounds that promote autophagy by modulating various cellular processes. While true CRMs that fully meet the required criteria are limited, there is a growing interest in identifying compounds with potential CRM properties. Here, we focus on generating novel molecules from CRMs, specifically focusing on metformin, a biguanide medication known for its ability to address age-related disorders and enhance overall health and lifespan, as metformin has gained significant attention and is currently being investigated in the clinical trial phase 4.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/AnshulVerma01/Novel-CRM-Generation-Pipeline.git
```
2. Create a conda environment:
```bash
conda env create -f CRM.yml
conda activate CRM
```

## Usage
The primary script to run the pipeline is provided src/mutAIverse.py. Execute it directly by changing the respective file paths.
```python
python metgen.py
```

Demographic of top 10 scorer of novel generated CRM molecules from Metformin
![Image of top 10 scorer generated molecules from Metformin](image/metgen.png)


**NOTE**: For running the pipeline, models for hallmarks of aging are required but we have not provided here yet as this project has not been published yet.
