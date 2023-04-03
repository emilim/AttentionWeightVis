# Attention weight visualization for Svelte

Visualize attention weights for a given input text. The model is trained on the ted_hrlr_translate/pt_to_en dataset and uses a transformer architecture. 

## Setting up the environment

First you will need to install the python dependencies. We recommend using a virtual environment. 
```bash
# install python dependencies
pip install -r requirements.txt
```
We use two python scripts, transformer.py and create_database.py, to create and train the model and to create the database based on the model's predictions and attention weights. 

The database consist of a Neo4j graph database.

The visualization is done using Svelte and a library called ForceGraph3D that you'll need to install. 

```bash
# install svelte dependencies
npm install

# develop with hot reload at localhost:5000
npm run dev

# build for production and launch server
npm run build
```