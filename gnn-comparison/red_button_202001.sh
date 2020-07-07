PREP_FOLDER=PREP_DATA_FOLDER
OUTER_K=10

mkdir DATA

# Chemical
py PrepareDatasets.py $PREP_FOLDER/CHEMICAL  --outer-k $OUTER_K --dataset-name PROTEINS
py PrepareDatasets.py $PREP_FOLDER/CHEMICAL  --outer-k $OUTER_K --dataset-name NCI1
py PrepareDatasets.py $PREP_FOLDER/CHEMICAL  --outer-k $OUTER_K --dataset-name ENZYMES
# Social - ones
py PrepareDatasets.py $PREP_FOLDER/SOCIAL_ONES  --outer-k $OUTER_K --dataset-name IMDB-BINARY --use-one
py PrepareDatasets.py $PREP_FOLDER/SOCIAL_ONES  --outer-k $OUTER_K --dataset-name IMDB-MULTI --use-one
py PrepareDatasets.py $PREP_FOLDER/SOCIAL_ONES  --outer-k $OUTER_K --dataset-name COLLAB --use-one

# Prep folders
cp -r $PREP_FOLDER/CHEMICAL/PROTEINS_full DATA/PROTEINS
cp -r $PREP_FOLDER/CHEMICAL/NCI1 DATA
cp -r $PREP_FOLDER/CHEMICAL/ENZYMES DATA
cp -r $PREP_FOLDER/SOCIAL_ONES/IMDB-BINARY DATA
cp -r $PREP_FOLDER/SOCIAL_ONES/IMDB-MULTI DATA
cp -r $PREP_FOLDER/SOCIAL_ONES/COLLAB DATA
mv $PREP_FOLDER ..

# Run comparison SOTA
PFX=5k_
DS=IMDB-BINARY
CUDA_VISIBLE_DEVICES=0 py Launch_Experiments.py --config-file config_GRNF_5k.yml --dataset-name $DS --result-folder $PFX$DS --debug
DS=IMDB-MULTI
CUDA_VISIBLE_DEVICES=0 py Launch_Experiments.py --config-file config_GRNF_5k.yml --dataset-name $DS --result-folder $PFX$DS --debug
DS=COLLAB
CUDA_VISIBLE_DEVICES=0 py Launch_Experiments.py --config-file config_GRNF_5k_cpu.yml --dataset-name $DS --result-folder $PFX$DS --debug
DS=NCI1
CUDA_VISIBLE_DEVICES=0 py Launch_Experiments.py --config-file config_GRNF_5k.yml --dataset-name $DS --result-folder $PFX$DS --debug
DS=ENZYMES
CUDA_VISIBLE_DEVICES=0 py Launch_Experiments.py --config-file config_GRNF_5k.yml --dataset-name $DS --result-folder $PFX$DS --debug
DS=PROTEINS
CUDA_VISIBLE_DEVICES=0 py Launch_Experiments.py --config-file config_GRNF_5k.yml --dataset-name $DS --result-folder $PFX$DS --debug

# Run convergence
MM=4 
CUDA_VISIBLE_DEVICES=0 py Launch_Experiments.py --config-file config_GRNF_varM$MM.yml --dataset-name IMDB-BINARY --result-folder varm_$MM --debug
MM=8 
CUDA_VISIBLE_DEVICES=0 py Launch_Experiments.py --config-file config_GRNF_varM$MM.yml --dataset-name IMDB-BINARY --result-folder varm_$MM --debug
MM=16 
CUDA_VISIBLE_DEVICES=0 py Launch_Experiments.py --config-file config_GRNF_varM$MM.yml --dataset-name IMDB-BINARY --result-folder varm_$MM --debug
MM=32 
CUDA_VISIBLE_DEVICES=0 py Launch_Experiments.py --config-file config_GRNF_varM$MM.yml --dataset-name IMDB-BINARY --result-folder varm_$MM --debug
MM=64 
CUDA_VISIBLE_DEVICES=0 py Launch_Experiments.py --config-file config_GRNF_varM$MM.yml --dataset-name IMDB-BINARY --result-folder varm_$MM --debug
MM=128 
CUDA_VISIBLE_DEVICES=0 py Launch_Experiments.py --config-file config_GRNF_varM$MM.yml --dataset-name IMDB-BINARY --result-folder varm_$MM --debug
MM=256 
CUDA_VISIBLE_DEVICES=0 py Launch_Experiments.py --config-file config_GRNF_varM$MM.yml --dataset-name IMDB-BINARY --result-folder varm_$MM --debug
MM=512 
CUDA_VISIBLE_DEVICES=0 py Launch_Experiments.py --config-file config_GRNF_varM$MM.yml --dataset-name IMDB-BINARY --result-folder varm_$MM --debug
MM=1024 
CUDA_VISIBLE_DEVICES=0 py Launch_Experiments.py --config-file config_GRNF_varM$MM.yml --dataset-name IMDB-BINARY --result-folder varm_$MM --debug
