# TODO 恐らくChat系の教師データは、モデルに合わせた形式にする必要があると思われる
class TrainingPromptGenerator:

    """チャット形式に対応する教師データを作る系のデータの場合はこちら"""
    DATA_TYPE_INSTRUCT_CHAT = "instruct"

    """プレーンテキストを教育して文書生成を行う場合はこちら"""
    DATA_TYPE_PLAINTEXT = "plaintext"

    @staticmethod
    def __get_instruct_training_data(data_point):
        if data_point["input"]:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

            ### Instruction:
            {data_point["instruction"]}

            ### Input:
            {data_point["input"]}

            ### Response:
            {data_point["output"]}"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

            ### Instruction:
            {data_point["instruction"]}

            ### Response:
            {data_point["output"]}"""

    @staticmethod
    def __get_instruct_prompt(prompt):
        if prompt["sub_prompt"]:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

            ### Instruction:
            {prompt["prompt"]}

            ### Input:
            {prompt["sub_prompt"]}

            ### Response:"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

            ### Instruction:
            {prompt["prompt"]}

            ### Response:"""

    @staticmethod
    def __get_plaintext_training_data(data_point):
        return data_point["text"]

    @staticmethod
    def __get_plaintext_prompt(prompt):
        return prompt["prompt"]

    @staticmethod
    def get_training_data(data_point, type):
        if type == TrainingPromptGenerator.DATA_TYPE_INSTRUCT_CHAT:
            return TrainingPromptGenerator.__get_instruct_training_data(data_point)
        elif type == TrainingPromptGenerator.DATA_TYPE_PLAINTEXT:
            return TrainingPromptGenerator.__get_plaintext_training_data(data_point)
        else:
            raise ValueError(f"未知のtypeが指定されました: {type}")

    @staticmethod
    def get_prompt(prompt, type):
        if type == TrainingPromptGenerator.DATA_TYPE_INSTRUCT_CHAT:
            return TrainingPromptGenerator.__get_instruct_prompt(prompt)
        elif type == TrainingPromptGenerator.DATA_TYPE_PLAINTEXT:
            return TrainingPromptGenerator.__get_plaintext_prompt(prompt)
        else:
            raise ValueError(f"未知のtypeが指定されました: {type}")
