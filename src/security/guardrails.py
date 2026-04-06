
class FinancialGuardrail:
    @staticmethod
    def validate_output(response_text: str):
        # Impede que o agente responda valores financeiros se não
        # for solicitado via Tool segura
        if "salário" in response_text.lower():
            # Lógica de máscara ou verificação de permissão
            pass
