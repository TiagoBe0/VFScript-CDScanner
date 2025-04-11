import csv
import ollama

class CSVInterpreter:
    def __init__(self, csv_file, model='deepseek-r1'):
       
        self.csv_file = csv_file
        self.model = model
        self.data = None

    def load_csv(self):
        try:
            with open(self.csv_file, newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                self.data = [row for row in reader]
        except Exception as e:
            print(f"Error al leer el archivo CSV: {e}")
            self.data = []

    def interpret(self, prompt=""):
        if self.data is None:
            self.load_csv()

        csv_content = "\n".join([", ".join(row) for row in self.data])

        message_content = f"{prompt}\n{csv_content}" if prompt else csv_content

        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": message_content}]
        )
        return response["messages"]["content"]

if __name__ == "__main__":

    interpreter = CSVInterpreter("outputs.vfinder/{}")
    resultado = interpreter.interpret(prompt="Interpreta los siguientes datos:")
    print(resultado)
