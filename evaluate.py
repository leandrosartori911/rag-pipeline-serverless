from openai import OpenAI
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall

client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

judge_llm = llm_factory("llama3.1:8b", provider="openai", client=client)
judge_embeddings = OpenAIEmbeddings(client=client, model="nomic-embed-text")

# Casos reais já testados nas Fases 4-6
test_cases = [
    {
        "user_input": "por que empresas de IA buscam países nórdicos para data centers?",
        "response": "Por causa do clima favorável e acesso direto a cabos transatlânticos de alta velocidade, o que reduz o tempo de resposta das aplicações de IA para os usuários finais ao redor do mundo.",
        "retrieved_contexts": [
            "O uso de cabos submarinos de fibra óptica é a principal espinha dorsal da internet global... Países nórdicos e ilhas no Atlântico Norte tornaram-se polos atraentes devido ao clima favorável e ao acesso direto a cabos transatlânticos de alta velocidade."
        ],
        "reference": "Empresas de IA buscam países nórdicos porque o processamento de modelos de linguagem exige resfriamento constante e grande consumo de energia, e essas regiões oferecem clima frio e fontes de energia renovável abundantes, além de acesso direto a cabos transatlânticos de alta velocidade."
    },
    {
        "user_input": "do que fala o documento?",
        "response": "Do direito das pessoas e proteção de dados.",
        "retrieved_contexts": [
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua..."
        ],
        "reference": "O documento não tem conteúdo real, é texto Lorem Ipsum sem significado (placeholder)."
    }
]

dataset = EvaluationDataset.from_list(test_cases)

results = evaluate(
    dataset=dataset,
    metrics=[
        Faithfulness(llm=judge_llm),
        LLMContextPrecisionWithReference(llm=judge_llm),
        LLMContextRecall(llm=judge_llm),
    ],
)

print(results)
df = results.to_pandas()
print(df.to_string())