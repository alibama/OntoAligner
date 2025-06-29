# Import necessary modules
from ontoaligner.ontology import CommonKGOntology
from ontoaligner import encoder
from ontoaligner.ontology_matchers import SimpleFuzzySMLightweight, SBERTRetrieval
from ontoaligner.postprocess import retriever_postprocessor
from ontoaligner.encoder import ConceptLLMEncoder
from ontoaligner.ontology_matchers import AutoModelDecoderLLM, ConceptLLMDataset
from ontoaligner.postprocess import TFIDFLabelMapper, llm_postprocessor

from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

# Step 1: Load and parse ontologies
ontology = CommonKGOntology()
parsed_ontology1 = ontology.parse("../assets/lexipedia/time.owl")  # Path to source ontology
parsed_ontology2 = ontology.parse("../assets/lexipedia/solidity.owl")     # Path to target ontology

# Step 2: Encode ontologies using a lightweight parent-based encoder
encoder_model = encoder.ConceptParentLightweightEncoder()
encoder_output = encoder_model(source=parsed_ontology1, target=parsed_ontology2)

# Debug: Check encoder output
print(f"Encoder output type: {type(encoder_output)}")
if hasattr(encoder_output, '__len__'):
    print(f"Encoder output length: {len(encoder_output)}")
if isinstance(encoder_output, list) and len(encoder_output) >= 2:
    source_data, target_data = encoder_output[0], encoder_output[1]
    print(f"Source ontology concepts: {len(source_data)}")
    print(f"Target ontology concepts: {len(target_data)}")
    
    # Check if either ontology is empty
    if len(source_data) == 0:
        print("WARNING: Source ontology is empty! Trying a different source file...")
        # Try parsing a different source ontology file that has content
        try:
            parsed_ontology1 = ontology.parse("solidity.owl")  # Use solidity as both source and target for demo
            encoder_output = encoder_model(source=parsed_ontology1, target=parsed_ontology2)
            source_data, target_data = encoder_output[0], encoder_output[1]
            print(f"After retry - Source ontology concepts: {len(source_data)}")
            print(f"After retry - Target ontology concepts: {len(target_data)}")
        except Exception as e:
            print(f"Error retrying with different source: {e}")
    
    if len(source_data) == 0 or len(target_data) == 0:
        print("ERROR: Cannot proceed with empty ontology data")
        exit(1)
        
print(f"Encoder output ready for matching")

# Step 3: Match using Simple Fuzzy String Matcher (Lightweight)
fuzzy_model = SimpleFuzzySMLightweight(fuzzy_sm_threshold=0.4)  # Set fuzzy string threshold
fuzzy_matchings = fuzzy_model.generate(input_data=encoder_output)

# Step 4: Print fuzzy matcher results
print("\n=== SimpleFuzzySMLightweight Matchings ===")
for match in fuzzy_matchings:
    print(f"Source: {match['source']}, Target: {match['target']}, Score: {match['score']}")

# Step 5: Match using SBERT Retriever
sbert_model = SBERTRetrieval(device="cpu", top_k=3)  # Top 3 candidates using SBERT
sbert_model.load(path="all-MiniLM-L6-v2")  # Load pre-trained model
sbert_matchings = sbert_model.generate(input_data=encoder_output)

# Step 6: Print SBERT matcher results
print("\n=== SBERTRetrieval Matchings ===")
for match in sbert_matchings:
    print(f"Source: {match['source']}, Target: {match['target-cands']}, Score: {match['score-cands']}")

# Step 7: Postprocess SBERT matchings (e.g., filter duplicates, normalize)
sbert_matchings = retriever_postprocessor(sbert_matchings)

# Step 8: Print postprocessed SBERT matchings
print("\n=== Post-Processed SBERT Matchings ===")
for match in sbert_matchings:
    print(f"Source: {match['source']}, Target: {match['target']}, Score: {match['score']}")

# Step 9: Encode ontologies for LLM-based matching
llm_encoder = ConceptLLMEncoder()
source_onto, target_onto = llm_encoder(source=parsed_ontology1, target=parsed_ontology2)

# Step 10: Prepare dataset for LLM decoder
llm_dataset = ConceptLLMDataset(source_onto=source_onto, target_onto=target_onto)

# Step 11: Create a DataLoader for batching LLM prompts
dataloader = DataLoader(
    llm_dataset,
    batch_size=2048,  # Batch size for LLM inference
    shuffle=False,
    collate_fn=llm_dataset.collate_fn
)

# Step 12: Load and configure the LLM decoder
# Use CPU and disable quantization for macOS compatibility
llm_model = AutoModelDecoderLLM(device="cpu", max_length=300, max_new_tokens=10)
llm_model.load(path="Qwen/Qwen2-0.5B")  # Load a small Qwen model

# Step 13: Generate predictions using LLM decoder
predictions = []
for batch in tqdm(dataloader, desc="Generating with LLM"):
    prompts = batch["prompts"]
    sequences = llm_model.generate(prompts)
    predictions.extend(sequences)

# Step 14: Postprocess LLM predictions using TF-IDF label mapper
mapper = TFIDFLabelMapper(
    classifier=LogisticRegression(),  # Classifier for label similarity
    ngram_range=(1, 1)                # Use unigram TF-IDF
)
llm_matchings = llm_postprocessor(predicts=predictions, mapper=mapper, dataset=llm_dataset)

# Step 15: Print final LLM-based matchings
print("\n=== LLM Matchings ===")
for match in llm_matchings:
    # Handle different possible formats for LLM matching results
    if isinstance(match, dict):
        source = match.get('source', 'N/A')
        target = match.get('target', 'N/A')
        score = match.get('score', match.get('confidence', 'N/A'))
        print(f"Source: {source}, Target: {target}, Score: {score}")
    else:
        print(f"LLM Match: {match}")
