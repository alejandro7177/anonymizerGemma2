import os, json
from datasets import Dataset

# Ruta donde están tus archivos JSON  
folder_path = f'{os.getcwd()}/dataset'

# Función para convertir cada archivo JSON al formato requerido
def convert_to_format(json_file):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # La entrada será el texto original
        input_text = data['text']
        
        # La salida será un JSON con el texto anonimizado y las entidades
        output_json = {
            "texto_anonimizado": data['text_anoni'],  # Cambié la clave a "texto_anonimizado"
            "entities": data['entities']
        }
        return {
            'instruction': """
            Anonymize the following text and extract entities. The labels should be in the following format: [person_n], where 'n' is the corresponding number. Ensure that the \"text_anoni\" field contains the full input text with every detected entity replaced by its corresponding [person_n] label. Do not modify any other parts of the text. Return the information in a JSON format as follows:\n\n{\"text_anoni\": \"<anonymized_text>\", \"entities\": [\n    {\"entity\": \"<extracted_entity>\", \"types\": [\"<entity_type>\"]}\n]}\n >
            EXAMPLE:
            ### Input:

            Capítulo IV **Análisis del régimen impugnativo de la Ley Nro. 7182 a la luz de la igualdad en general y de la igualdad procesal en particular** Mg. Agustín Badino 1. Introducción Como ya se insinuó en capítulos anteriores, debemos afirmar que la igualdad es uno de los elementos constitutivos del debido proceso adjetivo, máxime aun cuando lo trasponemos por el tamiz de la tutela judicial efectiva. En otras palabras, no podemos hablar de un debido proceso y, por consiguiente, de una efectiva tutela judicial, si su estructura no resguarda el principio rector de la igualdad. Ahora bien, la igualdad se trata de un instituto con distintivos difusos que reviste cierta complejidad
            
            ### Output:
            {
                "text_anoni": "Capítulo IV **Análisis del régimen impugnativo de la Ley Nro. 7182 a la luz de la igualdad en general y de la igualdad procesal en particular** Mg. [person_1] 1. Introducción Como ya se insinuó en capítulos anteriores, debemos afirmar que la igualdad es uno de los elementos constitutivos del debido proceso adjetivo, máxime aun cuando lo trasponemos por el tamiz de la tutela judicial efectiva. En otras palabras, no podemos hablar de un debido proceso y, por consiguiente, de una efectiva tutela judicial, si su estructura no resguarda el principio rector de la igualdad. Ahora bien, la igualdad se trata de un instituto con distintivos difusos que reviste cierta complejidad"
                "entities": [
                    {
                        "entity": "Agustín Badino",
                        "types": [
                            "person_1"
                        ]
                    }
                ]
            }
            """,
            'input': input_text,
            'output': json.dumps(output_json, ensure_ascii=False)  # Evita convertir caracteres a secuencias Unicode
        }
    except json.JSONDecodeError as e:
        print(f"Error al decodificar el archivo {json_file}: {e}")
        return None  # Devuelve None si el archivo tiene un error de formato
    except Exception as e:
        print(f"Error al procesar el archivo {json_file}: {e}")
        return None

# Leer todos los archivos JSON en la carpeta
all_data = []
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        result = convert_to_format(file_path)
        if result:  # Solo agregar si no hubo error
            all_data.append(result)

# Convertir la lista de diccionarios a un dataset de Hugging Face
dataset = Dataset.from_list(all_data)

# Función de formato para el modelo Unloth
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN ="<eos>"  # Asegúrate de que EOS_TOKEN esté definido

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Añadir EOS_TOKEN
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts }

# Mapear la función de formato a los datos
dataset = dataset.map(formatting_prompts_func, batched=True)

# Ahora tu dataset está listo para ser utilizado en el fine-tuning con Unloth.
# print(dataset[10]['text'])  # Para verificar el primer ejemplo del dataset

