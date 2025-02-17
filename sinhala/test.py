from muss.laser import get_laser_embeddings

# A sample file to test
sentences = open('/home/realtmxi/Github/muss/sinhala/MADLAD_CulturaX_cleaned/data/train-00003-of-00007.txt').readlines()[:10]
embeddings = get_laser_embeddings(sentences, language='')

print("Original Sentences:", sentences)
print("Tokenized/BPE Processed Sentences:", embeddings)
