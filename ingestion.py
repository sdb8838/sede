import os
from dotenv import load_dotenv
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_mistralai.embeddings import MistralAIEmbeddings
from mistralai import Mistral

load_dotenv()

# Logging Configuration
logging.basicConfig(level=logging.INFO)


MISTRAL_API_KEY=os.environ["MISTRAL_API_KEY"]
PERSIST_DIRECTORY="./.chroma"
COLLECTION_NAME="rag-chroma"
EMBEDDING_MODEL="mistral-embed"

mistral_client = Mistral(api_key=MISTRAL_API_KEY)
mistral_embeddings = MistralAIEmbeddings(model=EMBEDDING_MODEL, mistral_api_key=MISTRAL_API_KEY)


# Define function for creating and saving chroma vectorstore

def create_chroma_vectorstore(documents, embedding_function, persist_directory, collection_name):
    vectorstore = None # Initialize vectorstore to None
    try:
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_function,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
    except KeyError as ke:
        logging.error(f"KeyError occurred: {ke}")
        logging.error("Check MistralAI response and library compatibility.")

        if hasattr(embedding_function, "_endpoint_url"):
            response = embedding_function.client.post(
                embedding_function._endpoint_url, json={"input": documents}
            )
            logging.error(f"MistralAI API Response: {response.json()}")
        else:
            logging.error("Missing _endpoint_url attribute in MistralAIEmbeddings.")
    finally:
        # Check and return the vectorstore (whether it was created or not)
        return vectorstore


if __name__ == "__main__":
    plantilla_url = "https://sede.murcia.es/ficha-procedimiento?idioma=es&id="
    ids_procedimiento = [
        # Información y atención al ciudadano
        651, 221, 11, 201, 12,
        # Padrón de habitantes
        3662, 242, 301, 243, 302, 101,
        # Servicios sociales, sanidad, igualdad y cooperación
        # Cooperacción al desarrollo
        2782, 802, 3282,
        # Igualdad y atención a las víctimas de violencia de género
        5462, 4062,
        # Mayores y discapacidad
        3683, 3962,
        # Sanidad
        655, 656, 657, 665, 663, 5442, 5542, 658, 654, 662, 664,
        # Servicios sociales
        2962, 966, 2522, 3202, 965, 244, 4002, 3122, 362,
        # Urbanismo, vivienda y medio ambiente
        # Actividad económica
        2163, 2164, 2122, 2123, 2202, 2162, 2142, 2642, 2643, 2662, 2242, 2203, 3422, 2182, 2223, 2622, 2222,
        # Medio ambiente
        3262, 862, 883, 882, 842, 884, 864, 343, 863,
        # Urbanismo
        903, 694, 698, 2902, 730, 261, 5422, 262, 5642, 161, 564, 563, 565, 724, 3802, 3243, 3242, 2023, 263, 723, 342, 726,
        # Vivienda
        5302, 756, 5622, 757, 762, 758, 3682, 759, 760, 761, 3903,
        # Multas
        # Multas L.O. Seguridad ciudadana
        750,
        # Multas ordenanzas
        751, 745, 746, 747, 982, 1002, 748,
        # Multas de tráfico
        222, 755,
        # Tráfico y transportes
        # Oficina de la bicicleta
        765, 766,
        # Transportes
        504, 484, 2802, 485, 2822, 482, 481, 483,
        # Tráfico
        503, 501, 3882,
        # Actividades vía pública, parques y jardines
        # Vía pública
        4, 1902, 523, 542,
        # Cultura, Turismo, Juventud y Deportes
        # Cultura
        942,
        # Deportes
        3902, 3942, 1462, 1403, 381, 445, 364, 443, 1482,
        # Juventud
        461, 5162, 5582,
        # Turismo
        602,
        # Educación, formación y empleo
        # Educación
        822, 645, 644, 647, 648, 643, 646,
        # Empleo
        5182, 1562, 561, 562,
        # Participación ciudadana y descentralización
        # descentralización
        601,
        # Consumo, comercio y empresaa
        # Comercio
        # Empresas y autónomos
        3862, 4122, 5282,
        # Plazas y mercados
        584, 583, 586, 581, 569, 568, 585, 582,
        # Seguridad y emergencia
        # Extinción de incendios
        603,
        # Policía local
        567, 1224, 1225, 1223, 1222, 1226, 566,
        # Protección civil
        649, 650,
        # Hacienda, contratación y patrimonio
        # Consejo económico administrativo
        764,
        # Contratación
        # Patrimonio
        670, 671, 672, 673, 682, 684, 669, 685, 666, 686, 687, 689, 690, 683, 691, 667, 668, 693,
        # Planificación económica
        622,
        # Responsabilidad patrimonial
        782,
        # Tesorería
        3842, 3722,
        # Recursos humanos
        # Formación
        2842, 2862,
        # Procesos selectivos
        # Procesos selectivos. Promoción interna
        # provisión de puestos
        # Trámites personal Ayuntamiento de Murcia
        3402, 3362,
    ]
    urls = [plantilla_url + str(id_proc) for id_proc in ids_procedimiento]

    loader = WebBaseLoader(urls, continue_on_failure=False)
    docs=loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=2000, chunk_overlap=100
    )
    doc_splits = text_splitter.split_documents(docs)

    vectorstore = create_chroma_vectorstore(
        doc_splits, mistral_embeddings, PERSIST_DIRECTORY, COLLECTION_NAME
    )
