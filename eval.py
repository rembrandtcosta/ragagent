import time
from ragas import evaluate
from rag_workflow import process_question 
from ragas import EvaluationDataset
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas.llms import LangchainLLMWrapper

qa = [
  {
    "user_input": "Moro em um apartamento térreo, do tipo “garden”. Sofro com o barulho na área do roll de entrada. Os condôminos entram e saem gritando, batendo portas ou mantendo conversar longas em voz alta, tudo parece estar acontecendo dentro do meu apartamento. Já reclamei com a administração do condomínio e pedi apenas para que, nas assembleias, fosse dada uma orientação simples sobre isso, com o objetivo de conscientizar a todos sobre esse incomodo. Sugeri que colocassem placas pedindo silencio ou algo do tipo. A administradora e o sindico disseram que nada poderia ser feito e que as placas não adiantariam, pois é do brasileiro não respeitar nada.",
    "response": "Não concordo com as atitudes do Síndico e dos representantes da administradora. A lei é clara, o condômino ou seus dependentes, incluindo ainda o locatário, não podem deliberadamente praticarem atos que venham a por em risco o sossego dos demais moradores. É o próprio Código Civil, além de muitas convenções (a grande maioria delas) que veda tal comportamento e ainda prevê altas multas para quem reiteradamente gerar incompatibilidade de convivência (art. 1.337 parágrafo único)."
  },
  {
    "user_input": "Ao analisar o demonstrativo financeiro do mês de MAIO de 2016, observei que não foram lançados como receita os pagamentos de condomínio dos apartamentos, e este valor faltante não entrou nos meses subsequentes. O que devo fazer? (Agnaldo – Contagem)",
    "response": "Se a receita foi omitida, há de se suspeitar de “subtração de recursos”. O que deve ser feito é uma auditoria contábil para levantar possíveis irregularidades e apontar os responsáveis, que neste caso, quem primeiro responde é o síndico, já que cabe a ele tal atribuição (Art. 1.348, VII)."
  },
  {
    "user_input": "Minha amiga esqueceu de pagar o condomínio e agora a síndica quer receber todos os juros corridos na prestação do condomínio, ocorre que os juros são muito altos e a síndica não quer fazer acordo. Como posso auxiliá-la neste caso? (Tatiana – São Paulo)",
    "response": "Os juros e multa a qual estão sujeitos os condôminos que atrasam o pagamento de suas obrigações são aqueles previstos no Art. 1.336, I do Cód. Civil e são estipulados em 1% ao (os juros) e 2% a multa. Porém, este mesmo artigo permite que os condôminos convencionem outros percentuais, unicamente nos juros, permanecendo a multa em 2%. Assim, deve-se consultar a convenção do condomínio para se saber quais foram os juros convencionados, que, no silêncio, prevalecem os de 1% ao mês."
  },
  {
    "user_input": "Prédio edificado com três finalidades específicas: residencial, comercial e outra como centro clínico. Este igualmente com salas, cada uma com sua matrícula e proprietário. Cada “setor” tem seu próprio acesso à via pública estando isolados um do outro, não se utilizado de área comum para o acesso em cada “setor”, sendo a incorporação registrada como Condomínio Residencial e Comercial. Minha pergunta é: se é possível criar uma convenção de condomínio para cada “setor”, visto que são completamente independentes, apenas compõem o mesmo edifício? (Eduardo – São Leopoldo)",
    "response": "Uma vez havendo um único registro imobiliário do empreendimento, mesmo sendo ele de uso misto, não há do que se falar em multiplicidade de convenções, já que, a convenção é o ato institutivo do Condomínio (arts. 1.332 e 1.333 Cód. Civil). Neste caso o que pode ser adotado é um Regimento Interno específico para cada um dos “setores” deste Condomínio."
  }
]
qa = qa[:2]

dataset = []

for item in qa:
    query = item["user_input"]
    reference = item["response"]

    result = process_question(query)
    print(result)
    # time.sleep(25)
    relevant_docs = list(map(lambda doc: doc.page_content, result["documents"]))
    dataset.append(
        {
            "user_input": query,
            "retrieved_contexts": relevant_docs,
            "response": result["solution"],
            "reference": reference
        }
    )


evaluation_dataset = EvaluationDataset.from_list(dataset)

# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
llm = ChatOllama(model="llama3.2:1b", temperature=0)

evaluator_llm = LangchainLLMWrapper(llm)

result = evaluate(
    dataset=evaluation_dataset,
    metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
    llm=evaluator_llm
)

print(result)

