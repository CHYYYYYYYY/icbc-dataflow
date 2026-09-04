import random

from dataflow.core.prompt import PromptABC
from dataflow.prompts.tech_doc_prompt_snippets import (
    META_FORBIDDEN_ZH_CRITIQUE_BRIEF,
    META_FORBIDDEN_ZH_REFINE_BLOCK,
)
from dataflow.utils.registry import PROMPT_REGISTRY

'''
A collection of prompts for the general text operator.
'''

@PROMPT_REGISTRY.register()
class Phi4QAGeneratorPrompt(PromptABC):
    """
    # 用途：将文档段落转换为多轮对话格式（Question: Answer:）
    # 应用场景：在 Phi4QAGenerator 算子中使用，用于预训练数据生成
    #
    # 输入：原始文档内容
    # 输出格式：
    # Question: xxx Answer: xxx
    # Question: xxx Answer: xxx
    #
    # 特点：简单的格式转换，适合生成预训练格式的对话数据
    """
    def __init__(self):
        pass
    
    def build_prompt(self, content: str) -> str:
        """
        Generate the LLM input prompt by inserting the raw content into the prompt template.
        """
        prompt = """
        A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the questions. 
        Convert the following paragraph into a conversational format with multiple tags of "Question:" followed by "Answer:":

        You can only output as the given format:
        Question: xxx Answer: xxx
        Question: xxx Answer: xxx
        Now please covert the content below.
        {content}
        """
        return prompt.format(content=content)

@PROMPT_REGISTRY.register()    
class SFTGeneratorSeedPrompt(PromptABC):
    """
    # 用途：基于给定文档内容生成高质量的 SFT 训练数据
    # 应用场景：在 SFTGeneratorSeed 算子中使用
    #
    # 核心功能：
    # 1. 从文档中生成一个问答对
    # 2. 支持 custom_prompt 自定义额外要求
    # 3. 输出 JSON 格式：{"instruction": "问题", "output": "答案"}
    #
    # 特点：
    # - 支持自定义提示词（通过 custom_prompt 参数）
    # - 使用 <|im_start|> 对话格式
    # - 强调多样性、相关性和清晰度
    """
    def __init__(self, custom_prompt):
        self.custom_prompt = custom_prompt
        pass

    def build_prompt(self, content: str = "") -> str:
        """
        Generate the LLM input prompt by inserting the raw content into the prompt template,
        with optional custom instructions to enhance flexibility.
        """
        base_prompt = """You are tasked with creating high-quality SFT data for LLM training.
    Please generate one question based on the provided context, focusing on diversity, relevance, and clarity.

    Requirements:
    1. Generate exactly one distinct and well-formed question.
    2. The question must be based on the context and include enough background for clarity.
    3. Output must follow this JSON format:
    {{
        "instruction": "QUESTION",
        "output": "ANSWER"
    }}

    Examples:
    {{
        "instruction": "Can you provide a list of healthy habits to maintain a healthy lifestyle? Please format your response as an HTML page with bullet points.",
        "output": "Here's an HTML page with bullet points for healthy habits: <html><body><h3>Healthy Habits:</h3><ul><li>Eating a balanced diet...</li></ul></body></html>"
    }},
    {{
        "instruction": "How can we use Python to calculate the GCD (greatest common divisor) of five numbers and express each number in terms of the GCD?",
        "output": "Here's a Python function that calculates the GCD of five numbers: def find_gcd(...) ..."
    }}

    {custom_section}

    Now, based on the following context, please generate one question:
    """

        custom_section = f"Additional instruction:\n{self.custom_prompt}\n" if self.custom_prompt else ""
        full_prompt = base_prompt.format(custom_section=custom_section)
        
        return f"<|im_start|>system\n{full_prompt}<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant"


import textwrap

import textwrap

@PROMPT_REGISTRY.register()
class MetaPrompt(PromptABC):
    """
    # 用途：从6个自定义维度评估文本质量
    # 应用场景：在 MetaSampleEvaluator 算子中使用
    #
    # 工作机制：
    # 1. 初始化时传入 dimensions（维度列表）
    # 2. 每个维度包含：dimension_name、description、example_list
    # 3. 对文本进行多维度打分（1-5分）
    #
    # 输出格式：
    # - 每个维度一段评估分析
    # - 最后一行输出分数列表：[5, 4, 3, 5, 4, 5]
    #
    # 示例维度：
    # - Content Accuracy（内容准确性）
    # - Writing Quality（写作质量）
    # - Educational Value（教育价值）
    # 等等...
    #
    # 特点：高度可定制，可以根据需求定义任意6个评估维度
    """
    def __init__(self, dimensions):
        self.dimensions = self._format_dimensions(dimensions=dimensions)

        self.system_prompt_template = textwrap.dedent("""\
You are an expert evaluator of text content. You will be given a single piece of text and must evaluate it across six specific dimensions listed below. Each dimension includes a description and a list of concrete examples (example_list), each labeled with a quality score. Higher scores indicate better quality. Use these examples to guide your assessment.
{dimensions_list}

Instructions:
- Provide a clear evaluation for each of the six dimensions based on the input text.
- Each evaluation should be one short paragraph.
- Then assign an integer score from 1 to 5 for each dimension, where:
  5 = Excellent
  4 = Good
  3 = Fair
  2 = Poor
  1 = Very Poor

- Your output should end with a **separate final line** that contains a Python-style list of six integers in this format:
  [5, 4, 3, 5, 4, 5]
        """)

        self.user_prompt_template = textwrap.dedent("""\
            Please analyze and evaluate the following text:

Text:
{text}

Your output should include:
- One paragraph of analysis for each of the six quality dimensions listed above.
- A final line with your scores in this exact format:
  [score1, score2, score3, score4, score5, score6]
        """)
        
    def _format_dimensions(self, dimensions):
        formatted_list = []

        for i, item in enumerate(dimensions, 1):
            
            examples_str = "\n".join([
                f'Example (Score: {ex["score"]}):\n"{ex["text"]}"\n'
                for ex in item["example_list"]
            ])
            block = f"""\"\"\"{i}. {item["dimension_name"]}: {item["description"]}

{examples_str}\"\"\""""
            formatted_list.append(block)
        return formatted_list


    def build_system_prompt(self):
        dimensions_text = "\n".join(self.dimensions)
        return self.system_prompt_template.format(dimensions_list=dimensions_text)

    def build_prompt(self, text):
        return self.user_prompt_template.format(text=text)

@PROMPT_REGISTRY.register()
class AlpagasusPrompt(PromptABC):
    """
    # 用途：评估 AI 助手对指令的响应质量
    # 应用场景：在 AlpagasusFilter 算子中使用，过滤低质量 SFT 数据
    #
    # 评估维度：可自定义（默认为 'quality'）
    # 评分范围：0-5分
    #
    # 输入：
    # - instruction：用户指令
    # - input：输入内容
    # - response：AI 响应
    #
    # 输出格式：
    # 第一行：分数（0-5）
    # 后续行：详细评估解释
    #
    # 特点：基于 Alpagasus 论文的数据过滤方法
    """
    def __init__(self, dimension='quality'):
        self.dimension = dimension
        self.system_prompt_template = """
        We would like to request your feedback on the performance of AI assistant in response to the instruction and the given input displayed following.
        Instruction: {instruction}
        Input: {input}
        Response: {response}
        """
        self.user_prompt_template = """
        Please rate according to the {dimension} of the response to the instruction and the input. Each assistant
        receives a score on a scale of 0 to 5, where a higher score indicates a higher level of the {dimension}. Please
        first output a single line containing the value indicating the scores. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias.
        """

    def build_system_prompt(self, instruction, input_text, response):
        """
        生成system prompt
        """
        return self.system_prompt_template.format(instruction=instruction, input=input_text, response=response)

    def build_prompt(self):
        """
        生成user prompt
        """
        return self.user_prompt_template.format(dimension=self.dimension)

@PROMPT_REGISTRY.register()
class TreeinstructPrompt(PromptABC):
    """
    # 用途：通过语义解析将指令转换为树结构，计算节点数量来衡量复杂度
    # 应用场景：评估指令的复杂程度
    #
    # 工作流程：
    # 1. 将指令解析为树结构（TREE-1）
    # 2. 计算树的节点数量
    # 3. 节点数量代表指令的复杂度
    #
    # 输出格式：
    # 最后一行只输出一个数字，例如：4
    #
    # 用途：可用于筛选不同复杂度的指令，或平衡数据集的难度分布
    """
    def __init__(self):
        self.system_prompt_template = """
        You are an instruction rewriter. You need to parse a given user instruction into a TREE structure following Semantic Parsing in the natural language processing field.
        Procedure:
        step-1: Parse the old “instruction” to a TREE-1 through Semantic Parsing in the natural language processing field. 
        Count and return the number of nodes in TREE-1.
        Old instruction: “{instruction}”
        """

        self.user_prompt_template = """
        Please count and return the number of nodes in TREE-1. This number represents the complexity of the original instruction.
        Output the number in the single LAST line. You must ensure the last line is only the number of the tree, without other symbols, like ```.
        For example:
        4
        """
    
    def build_system_prompt(self, instruction):
        """
        根据给定的指令生成 system prompt
        """
        return self.system_prompt_template.format(instruction=instruction)
    
    def build_prompt(self):
        """
        生成 user prompt
        """
        return self.user_prompt_template


@PROMPT_REGISTRY.register()
class ConsistentQueryPrompt(PromptABC):
    """
    # 用途：生成具有主题一致性的多轮对话问题
    # 应用场景：在 ConsistentChatGenerator 算子中使用，生成多轮对话数据
    #
    # 核心特点：
    # 1. 预定义了9大交互类型（Problem Solving、Educational、Health等）
    # 2. 每个类型包含多个信息流模式（info_flow）
    # 3. 每个类型包含大量预定义主题（topic_dict）
    #
    # 生成逻辑：
    # - 随机选择交互类型 → 随机选择信息流 → 随机选择主题
    # - 生成6-8轮连贯的用户问题
    #
    # 输出格式：
    # {
    #   "category": "核心主题",
    #   "turns": ["问题1", "问题2", "问题3", ...]
    # }
    #
    # 特点：
    # - 问题自然、口语化
    # - 避免过于机械的表达
    # - 保持主题一致性
    """
    def __init__(self):
        self.intent_categories = {
            "Problem Solving Interaction": [
                "From Problem Diagnosis to Solution Optimization"
            ],
            "Educational Interaction": [
                "From Broad Theory to Specific Scenarios",
                "From Basic Concepts to Cross-Domain Connections"
            ],
            "Health Consultation Interaction": [
                "From Problem Diagnosis to Solution Optimization",
                "From Hypothesis Testing to Substantive Discussion"
            ],
            "Exploratory Interaction": [
                "From Time Sequence Expansion to Explore Causes and Effects",
                "From Hypothesis Testing to Substantive Discussion"
            ],
            "Entertainment Interaction": [
                "From Single Perspective to Multiple Perspectives",
                "From Hypothesis Testing to Substantive Discussion"
            ],
            "Simulation Interaction": [
                "From User Needs to Solutions",
                "From Broad Theory to Specific Scenarios"
            ],
            "Emotional Support Interaction": [
                "From Single Perspective to Multiple Perspectives",
                "From User Needs to Solutions"
            ],
            "Information Retrieval Interaction": [
                "From Basic Concepts to Cross-Domain Connections",
                "From Time Sequence Expansion to Explore Causes and Effects"
            ],
            "Transaction Interaction": [
                "From User Needs to Solutions",
                "From Problem Diagnosis to Solution Optimization"
            ]
        }
        self.topic_dict = {
            "Problem Solving Interaction": [
                "Technical support for computer hardware issues",
                "Home repair advice for plumbing problems",
                "Planning a budget-friendly vacation",
                "Fixing issues with internet connectivity",
                "Setting up a smart home system",
                "Solving problems with a broken washing machine",
                "Troubleshooting a malfunctioning printer",
                "How to repair a car engine",
                "Fixing a cracked phone screen",
                "Troubleshooting Wi-Fi network issues",
                "Diagnosing problems with a non-responsive remote control",
                "How to reset a frozen smartphone",
                "Dealing with an overheating laptop",
                "Replacing a broken laptop screen",
                "How to upgrade computer RAM",
                "Fixing a leaking faucet",
                "How to unclog a kitchen sink",
                "Diagnosing a noisy refrigerator",
                "How to seal window drafts",
                "Troubleshooting a non-working ceiling fan",
                "Setting up a home office on a budget",
                "Fixing a car that won’t start in cold weather",
                "How to troubleshoot GPS navigation issues",
                "Fixing problems with a garage door opener",
                "Troubleshooting smart light bulbs that won’t connect",
                "Replacing a broken door lock",
                "Fixing a noisy air conditioning unit",
                "Troubleshooting camera connectivity on a laptop",
                "How to repair a broken headphone jack",
                "Setting up a secure home Wi-Fi network",
                "Replacing a smartphone battery",
                "Installing a wall-mounted TV safely",
                "Calibrating a smart thermostat",
                "Fixing screen flickering on a monitor",
                "Diagnosing strange noises from a desktop computer",
                "Solving Bluetooth connection problems",
                "Repairing a jammed paper shredder",
                "Troubleshooting slow smartphone performance",
                "How to stop water leakage under a bathroom sink",
                "Installing weather stripping on doors",
                "Setting up parental controls on a router",
                "Fixing a dishwasher that won’t drain",
                "Repairing a damaged phone charging port",
                "Replacing a worn-out windshield wiper",
                "How to fix a garage light that keeps flickering",
                "Solving battery drain issues in electric vehicles",
                "Resetting a smart TV to factory settings",
                "Troubleshooting a wireless keyboard that won't connect",
                "How to install a backup camera in a car"
            ],
            "Educational Interaction": [
                "Learning a new language online",
                "Understanding the basics of physics",
                "Music theory and basic chord progressions",
                "The basics of machine learning and AI",
                "Introduction to computer programming",
                "Understanding the structure of DNA",
                "Exploring the history of the Roman Empire",
                "The principles of economics",
                "The process of photosynthesis in plants",
                "Studying the human circulatory system",
                "Learning algebra and solving equations",
                "The basics of chemistry and atomic structure",
                "Studying world geography",
                "Learning about climate change and sustainability",
                "Understanding how the internet works",
                "Intro to creative writing techniques",
                "Basics of digital photography",
                "Understanding historical timelines",
                "Learning financial literacy and budgeting",
                "Exploring different art movements",
                "Understanding gravity and Newton’s laws",
                "Learning HTML and CSS for web design",
                "Exploring the solar system",
                "Basics of environmental science",
                "Introduction to statistics",
                "Learning about the American Civil War",
                "Understanding cultural anthropology",
                "Exploring human anatomy",
                "Learning basic sign language",
                "Intro to public speaking skills",
                "Introduction to ethical philosophy",
                "Learning how to conduct scientific experiments",
                "Studying global political systems",
                "Understanding basic genetics and heredity",
                "Learning how to analyze literature",
                "Basics of entrepreneurship and starting a business",
                "Studying ancient civilizations like Mesopotamia and Egypt",
                "Introduction to psychology and behavior",
                "Basics of digital citizenship and online safety",
                "Understanding the water cycle and weather patterns",
                "Learning how to write a research paper",
                "Studying global religions and belief systems",
                "Intro to logic and critical thinking",
                "Understanding supply and demand in markets",
                "Learning spreadsheet skills (e.g., Excel or Google Sheets)",
                "Introduction to cybersecurity principles",
                "Understanding different learning styles",
                "Basics of health and nutrition science",
                "Learning how to debate effectively"
            ],

            "Health Consultation Interaction": [
                "Tips for maintaining a healthy diet",
                "Analyzing symptoms of the common cold",
                "Dealing with seasonal allergies",
                "Understanding mental health and depression",
                "Health benefits of regular exercise",
                "Managing high blood pressure",
                "Identifying signs of anxiety disorder",
                "Dealing with insomnia and sleep problems",
                "Coping with stress in the workplace",
                "Understanding the impact of smoking on health",
                "Preventing type 2 diabetes through lifestyle changes",
                "Dealing with chronic back pain at home",
                "How to support immune health naturally",
                "Recognizing early signs of dehydration",
                "Understanding the effects of caffeine on the body",
                "Managing cholesterol through diet",
                "How to build a sustainable workout routine",
                "Mental health tips for remote workers",
                "Safe exercises for people with joint pain",
                "How to talk to a doctor about personal health concerns",
                "Advice for managing menstrual cramps",
                "Tips for healthy weight loss",
                "Understanding the role of sleep in mental wellness",
                "How to identify food intolerances",
                "Preventing common sports injuries",
                "Maintaining good posture while working",
                "Recognizing early signs of burnout",
                "How to manage asthma symptoms",
                "The importance of hydration for brain function",
                "Understanding the risks of sedentary lifestyles",
                "Managing digestive issues like bloating or IBS",
                "How to support bone health as you age",
                "Tips for quitting alcohol or reducing intake",
                "Understanding the benefits of mindfulness and meditation",
                "Recognizing signs of vitamin deficiency",
                "Safe stretching routines for flexibility",
                "How to create a balanced meal plan",
                "Managing migraines and chronic headaches",
                "Supporting eye health in the digital age",
                "Understanding how hormones affect mood and health",
                "Caring for skin during seasonal changes",
                "Understanding the basics of reproductive health",
                "Dealing with minor injuries at home (cuts, sprains)",
                "Tips for building mental resilience",
                "Creating a daily self-care routine",
                "Navigating food labels and nutrition facts",
                "Identifying signs of eating disorders",
                "How to stay active while traveling",
                "The role of gut health in overall wellness"
            ],
            "Exploratory Interaction": [
                "Exploring the concept of time travel",
                "Deep-sea exploration and underwater ecosystems",
                "Historical events that shaped the world",
                "The impact of artificial intelligence on society",
                "Exploring the mysteries of the Bermuda Triangle",
                "Investigating space exploration and Mars missions",
                "The history of human migration",
                "The future of renewable energy",
                "The impact of global warming on biodiversity",
                "Exploring the ancient pyramids of Egypt",
                "Uncovering the secrets of black holes",
                "The cultural significance of ancient myths",
                "Exploring parallel universes and multiverse theories",
                "The origins and evolution of language",
                "How ancient civilizations built megastructures",
                "The search for extraterrestrial life",
                "How volcanoes have shaped Earth’s surface",
                "The psychology of dreams and their meanings",
                "The science behind natural disasters",
                "Exploring the concept of simulated reality",
                "How ancient trade routes influenced global development",
                "Exploring lost civilizations and archaeological mysteries",
                "The evolution of the internet and digital culture",
                "How pandemics have influenced human history",
                "The ethics of genetic modification",
                "Exploring the possibility of underwater cities",
                "How cultural identity evolves through migration",
                "The role of philosophy in modern science",
                "Unsolved mysteries in astrophysics",
                "Exploring ancient astronomical observatories",
                "The influence of mythologies on modern storytelling",
                "How ancient weather patterns affected human settlement",
                "Exploring the idea of colonizing other planets",
                "The rise and fall of legendary empires",
                "The possibility of time dilation in deep space travel",
                "The influence of alchemy on early science",
                "Understanding cryptids and mythological creatures",
                "Exploring the legends of Atlantis",
                "How music evolved across civilizations",
                "The significance of sacred geometry in ancient structures",
                "How ancient calendars predicted celestial events",
                "The philosophy of consciousness and existence",
                "Exploring the science behind telepathy and ESP",
                "The history of espionage and intelligence gathering",
                "How plagues transformed the course of empires",
                "The psychology behind conspiracy theories",
                "Exploring the idea of digital immortality",
                "How ancient seafaring changed the world map",
                "The role of chaos theory in understanding the universe"
            ],
            "Entertainment Interaction": [
                "Creating a video game character",
                "Writing a mystery novel",
                "Designing a new board game",
                "Exploring a new fantasy world in literature",
                "The psychology behind horror movies",
                "The evolution of action films",
                "Playing a strategic card game",
                "Exploring the art of stand-up comedy",
                "How to produce an indie film",
                "Creating an engaging video game storyline",
                "Writing a screenplay for a short film",
                "Building a fantasy football team",
                "Exploring behind-the-scenes movie production",
                "Learning the basics of animation",
                "Creating your own comic book series",
                "Composing an original song",
                "Understanding character arcs in drama series",
                "Creating a YouTube channel for entertainment",
                "Developing a murder mystery dinner party game",
                "Exploring cosplay and costume design",
                "Designing the rules for a role-playing game",
                "Recording a podcast about pop culture",
                "Writing a fan fiction story",
                "Creating a music video on a budget",
                "Directing a scene with amateur actors",
                "Exploring live streaming as an entertainer",
                "Hosting an online trivia night",
                "Analyzing what makes a sitcom successful",
                "Creating viral content for social media",
                "Building a digital art portfolio for entertainment",
                "Learning how to voice act for animations or games",
                "Creating an interactive story with branching choices",
                "Reviewing and critiquing movies or TV shows",
                "Designing merchandise for a fictional brand",
                "Building a fictional world map for a fantasy series",
                "Creating theme music for a character or story",
                "Learning stage acting vs. screen acting",
                "Writing and performing a comedy skit",
                "Planning a virtual concert or talent show",
                "Designing a puzzle game with narrative elements",
                "Writing a parody song or video",
                "Hosting a fictional radio show",
                "Analyzing storytelling techniques in video games",
                "Developing an ARG (Alternate Reality Game)",
                "Creating concept art for a fantasy setting",
                "Writing dialogue for an animated series",
                "Planning a short film festival with friends",
                "Exploring sound design for entertainment media",
                "Building a fan community around fictional works"
            ],
            "Simulation Interaction": [
                "Business negotiations and decision-making",
                "Military strategy and planning simulations",
                "Simulation for emergency disaster response",
                "Flight training using simulators",
                "Healthcare simulation for medical professionals",
                "Simulating financial market crashes",
                "Simulating environmental disaster scenarios",
                "Running a simulated space mission",
                "Simulating customer service interactions",
                "Creating a disaster management simulation game",
                "Simulating a day in the life of a CEO",
                "Virtual reality driving test training",
                "Crisis management simulation for public relations",
                "Political campaign simulation and voter behavior",
                "Simulating ethical dilemmas in AI development",
                "Simulating the spread of infectious diseases",
                "Urban planning simulation for smart cities",
                "Simulating climate change over 100 years",
                "Training simulations for cybersecurity breaches",
                "Economic policy decision-making simulation",
                "Simulating courtroom trials and legal strategy",
                "Simulation for emergency room triage",
                "Virtual surgery practice for medical students",
                "Simulating supply chain disruptions",
                "Simulating archaeological digs and discoveries",
                "Spacewalk training in zero-gravity simulation",
                "Language learning through role-playing simulation",
                "Simulating diplomatic negotiations between countries",
                "Astronaut survival training simulation",
                "Simulating startup business pitch competitions",
                "Simulating historical battles for education",
                "Virtual restaurant management and customer flow simulation",
                "Simulating the effects of social media algorithms",
                "Driving public transportation in urban simulations",
                "Simulating a courtroom debate in a mock trial",
                "Disaster recovery planning for IT infrastructure",
                "Simulating election outcomes based on real-time data",
                "Simulation of water resource management in agriculture",
                "Creating a theme park operations simulator",
                "Simulating robotics navigation in dynamic environments",
                "Simulated coaching for sports teams",
                "Simulating ethical decision-making in journalism",
                "Simulating airport ground operations and logistics",
                "Simulating the development of a new pharmaceutical drug",
                "Simulated investment portfolio risk management",
                "Simulating refugee crisis response scenarios",
                "Virtual museum curation and exhibition planning",
                "Simulating interpersonal communication in therapy sessions",
                "Simulating AI behavior in self-driving vehicles",
                "Virtual internship simulation for workplace readiness"
            ],

            "Emotional Support Interaction": [
                "Coping with the death of a loved one",
                "Supporting a friend through a breakup",
                "Dealing with feelings of loneliness",
                "Coping with stress and work-life balance",
                "Managing anxiety during uncertain times",
                "Dealing with feelings of inadequacy",
                "Supporting someone going through mental health challenges",
                "Building resilience after a setback",
                "Managing anger and frustration",
                "Finding emotional support after a major life change",
                "Handling the emotional impact of job loss",
                "Coping with social anxiety in group settings",
                "Dealing with the fear of failure",
                "Recovering from a toxic relationship",
                "Supporting a child through emotional distress",
                "Dealing with homesickness when living abroad",
                "Finding motivation during depressive episodes",
                "Coping with a chronic illness diagnosis",
                "Navigating emotional burnout as a caregiver",
                "Overcoming feelings of rejection",
                "Learning to forgive yourself after a mistake",
                "Supporting a partner dealing with trauma",
                "Handling the emotions of being a new parent",
                "Rebuilding confidence after public embarrassment",
                "Managing expectations during major life transitions",
                "Dealing with guilt from past decisions",
                "Helping someone through a panic attack",
                "Coping with grief after a pet passes away",
                "Facing loneliness during the holiday season",
                "Balancing emotional vulnerability and self-protection",
                "Processing emotions after a traumatic event",
                "Helping teens deal with peer pressure",
                "Managing jealousy in a relationship",
                "Supporting an elderly parent with emotional needs",
                "Navigating friendship breakups with maturity",
                "Coping with fear of the future",
                "Dealing with body image issues and self-worth",
                "Handling emotional distance in long-term relationships",
                "Managing stress related to academic pressure",
                "Providing comfort to someone experiencing shame",
                "Processing mixed emotions after a big achievement",
                "Supporting someone with PTSD triggers",
                "Coping with infertility and emotional distress",
                "Rebuilding trust after betrayal",
                "Helping a loved one experiencing suicidal thoughts",
                "Dealing with emotional triggers in daily life",
                "Finding peace with an unresolved conflict",
                "Managing emotions after relocation or immigration",
                "Coping with fear of abandonment"
            ],
            "Information Retrieval Interaction": [
                "Finding the best tech product reviews online",
                "Looking up information on the latest scientific discoveries",
                "How to find reliable health advice on the internet",
                "Searching for a vacation destination based on reviews",
                "Finding the most recent climate change data",
                "Looking for historical documents on ancient civilizations",
                "Researching news about artificial intelligence advancements",
                "Finding user reviews for a new gadget",
                "Searching for scholarly articles on quantum computing",
                "Finding government reports on public health",
                "Locating top-rated online courses for career development",
                "Finding official information on visa requirements",
                "Researching the latest trends in the stock market",
                "Finding statistical data for academic research",
                "Looking up real-time traffic and commute updates",
                "Finding reviews and ratings for local restaurants",
                "Searching for housing market reports in a specific city",
                "Finding information on upcoming local events",
                "Researching criminal records or public legal cases",
                "Finding comparison data on different insurance policies",
                "Searching for open-source software alternatives",
                "Looking up case studies for business or marketing",
                "Finding details on government aid programs",
                "Researching side effects of prescription medications",
                "Finding technical documentation for programming libraries",
                "Looking up airline safety records",
                "Searching for consumer complaint databases",
                "Finding educational videos on historical topics",
                "Researching the genealogy of a family name",
                "Looking up employment law information by state",
                "Finding patent information for a new invention",
                "Researching cultural practices in different countries",
                "Searching for reviews of online learning platforms",
                "Finding data on renewable energy usage by country",
                "Looking up public records on local property ownership",
                "Finding historical weather data for a location",
                "Searching for quotes and citations in classic literature",
                "Finding nutrition information for restaurant meals",
                "Researching ethical sourcing of fashion brands",
                "Looking up vehicle recall history by VIN",
                "Finding demographic data for a specific region",
                "Researching nonprofit organization transparency reports",
                "Searching for academic conference proceedings",
                "Finding ratings and reviews of mobile apps",
                "Looking up historical election results by district",
                "Finding documentation on space exploration missions",
                "Researching funding opportunities for small businesses",
                "Searching for media coverage on social justice issues",
                "Finding open data sets for machine learning training",
                "Looking up safety information on household chemicals"
            ],
            "Transaction Interaction": [
                "Booking a flight online for a vacation",
                "How to purchase concert tickets online",
                "Making an appointment with a service provider",
                "Ordering food online for delivery",
                "Purchasing a product through an e-commerce site",
                "How to buy insurance online",
                "Scheduling a medical appointment",
                "Making a donation to a charity online",
                "Buying a gift card for a friend",
                "How to apply for a mortgage loan",
                "Renewing a vehicle registration online",
                "Paying utility bills through a mobile app",
                "Booking a hotel room for a weekend trip",
                "Registering for an online course or certification",
                "Subscribing to a streaming service",
                "Buying event tickets with a digital wallet",
                "Applying for a credit card through a website",
                "Reserving a rental car at the airport",
                "Paying property taxes online",
                "Purchasing digital books or audiobooks",
                "Ordering groceries from an online supermarket",
                "Paying tuition fees through a university portal",
                "Signing up for a gym membership online",
                "Applying for unemployment benefits digitally",
                "Reserving a table at a restaurant using an app",
                "Buying and downloading software securely",
                "Sending money internationally via online banking",
                "Registering a domain and hosting a website",
                "Buying stocks or cryptocurrency through a trading platform",
                "Purchasing travel insurance before a trip",
                "Ordering custom clothing or merchandise online",
                "Buying a used car through an online marketplace",
                "Paying for public transportation with a mobile wallet",
                "Subscribing to a monthly subscription box service",
                "Purchasing online advertising for a small business",
                "Topping up a prepaid phone plan online",
                "Paying for freelance services via a gig platform",
                "Placing a mobile order for in-store pickup",
                "Applying for a personal loan through a fintech app",
                "Booking a guided tour or local experience online",
                "Paying entry fees for a virtual event or webinar",
                "Setting up automatic payments for monthly bills",
                "Buying furniture or home goods with financing options",
                "Purchasing digital game content or in-app items",
                "Contributing to a crowdfunding campaign",
                "Paying for parking through a mobile parking app",
                "Ordering prescription medication online",
                "Reserving coworking space for remote work",
                "Paying for tutoring or online lessons"
            ]
        }
    
    def build_prompt(self, num_dialogs_per_intent):
        prompt = """
        Task Description and Rules 
        1. Generate multiple rounds of realistic user questions based on the provided topic: 
        - Based on a single core topic (provided directly by the user), generate multiple rounds of realistic user questions, comprising 6-8 turns in total. 
        - The questions should match the characteristics of real users in natural communication: sometimes simple, sometimes vague, or including contextual backgrounds, and should reflect the language style of daily communication. 
        - Note: Avoid directly including the exact expression of the input topic in the questions. Instead, abstract it with natural and conversational language in practical scenarios. 
        
        2. Dynamic Dialogue Information Flow in Conversations: Below are the relevant steps of the information flow: {info_flow}

        The dialogue style should adhere to the following requirements: 
        - Utilize natural phrasing and vivid language, avoiding overly mechanical responses. 
        - Favor shorter sentences in questions, with occasional subject omission allowed. 
        - Ensure smooth and logical transitions through lighthearted or entertaining interjections. 
        - Permit the expression of specific personality traits and individualized tones. 
        - Proactively introduce new topics when appropriate, ensuring relevance to the current theme. 
        
        The dialogue should comply with the following generation rules: 
        - For each round of dialogue, only simulate user questions without providing answers. 
        - Ensure the conversation flows naturally and reflects realistic interactive thinking. 
        - Avoid overly polished or templated content, ensuring the questions feel authentic and relatable in life scenarios. 
        
        Output Format: 
        Multi-turn Questions in JSON Format: 
        "category": "<Core Topic of the Conversation>", 
        "turns": ["<turn_1>", "<turn_2>", "<turn_3>", "..."] 
        To generate multi-turn queries with high topic consistency, please think step-by-step. 
        The input core topic for this task is: {topic}
        """
        all_query_prompts = []
        for intent, info_flows in self.intent_categories.items():
            for _ in range(num_dialogs_per_intent):
                info_flow = random.choice(info_flows)
                topic = random.choice(self.topic_dict[intent])
                query_prompt = prompt.format(info_flow=info_flow, topic=topic)
                all_query_prompts.append(query_prompt)
        return all_query_prompts


@PROMPT_REGISTRY.register()
class ConsistentResponsePrompt(PromptABC):
    """
    # 用途：为 ConsistentQueryPrompt 生成的问题生成连贯的回答
    # 应用场景：在 ConsistentChatGenerator 算子中使用
    #
    # 输入：
    # - topic：对话主题
    # - queries：问题列表
    #
    # 生成要求：
    # 1. 直接回答当前问题，不提出额外问题
    # 2. 保持上下文一致性
    # 3. 提供可操作的建议和解决方案
    # 4. 自然、支持性的语气
    #
    # 特点：
    # - 避免不必要的反问
    # - 关注实用性和可操作性
    # - 保持对话的逻辑连贯性
    """
    def __init__(self):
        pass
    
    def build_prompt(self, topic, queries):
        prompt = f"""
        Your task is to simulate a multi-turn conversation where you progressively answer a series of user questions provided under a given topic category. For each answer, focus on delivering a natural, contextually relevant, and actionable response while considering both the current question and future questions in the sequence. The goal is to ensure consistency and logical progression throughout the dialogue and to avoid unnecessary follow-up questions in the responses simultaneously. To generate multi-turn responses with high topic consistency, think step-by-step. Key Dialogue Style Requirements are as follows: 
        Content and Structure:
        1. Directly Answer the Current Question:
        - Provide a complete, useful response to the current question without posing additional questions unless they are directly relevant to future queries. 
        - If clarification or additional steps are needed, frame these as suggestions or explanations rather than questions.
        2. Be Context-Aware:
        - Always tailor each response to the current question while remaining mindful of the context provided by prior and future questions.
        - Avoid prematurely addressing future queries but create subtle links where necessary to ensure smooth progression.
        3. Clear, Action-Oriented Responses:
        - Focus on providing actionable advice, logical explanations, or troubleshooting steps rather than speculative or rhetorical remarks.
        - Avoid long or overly complex explanations; aim for clarity and efficiency.
        Tone and Style:
        1. Conversational and Supportive:
        - Use a natural, empathetic tone that simulates real-life problem-solving interactions.
        - Avoid mechanical or overly formal responses.
        2. Economical with Words:
        - Keep responses concise but informative. Minimize extraneous content while ensuring answers have enough detail to be helpful.
        3. No Unnecessary Questions:
        - Limit unnecessary questions in the responses and focus instead on providing actionable steps or solutions directly. Avoid follow-up questions that don’t align with the next user query.
        Turn-by-Turn Instructions:
        1. Answer Exclusively for the Current Question:
        - For each turn, generate an answer that directly addresses the immediate question. Avoid revisiting past details unnecessarily unless they are highly relevant.
        - While you shouldn’t anticipate or directly answer future queries, your response should create natural openings for upcoming questions if applicable.
        2. Avoid Irrelevant Follow-Up Questions:
        - If the immediate question doesn’t require clarification, frame your response as a statement or suggestion rather than a question.
        - Maintain alignment with the logical flow of dialogue to ensure each turn is coherent.
        3. Proactively Provide Scenarios or Steps:
        - Where appropriate, guide the user with specific recommendations, troubleshooting actions, or observations they can make without requiring back-and-forth clarification.
        Output Requirements:
        The output must simulate the conversation by only providing responses (one per turn) in a sequential manner. The final format must strictly adhere to valid JSON and include the required structure.
        
        The input core topic and questions-only turns for this task is: 
        core topic: {topic}
        queries:
        {', '.join([f'User query: {query}' for query in queries])}
        """
        return prompt
    
@PROMPT_REGISTRY.register()
class CondorQuestionPrompt(PromptABC):
    """
    # 用途：基于主题和领域生成高质量的 SFT 问题
    # 应用场景：在 CondorGenerator 算子中使用
    #
    # 核心特点：
    # 1. 预定义了大量主题分类（Marriage、Entertainment、AI、Healthcare等）
    # 2. 每个主题下有多个子领域
    # 3. 每个子领域有具体的标签
    #
    # 生成要求：
    # - 一次生成3个不同难度的问题
    # - 问题必须有上下文和充分信息
    # - 不能突兀地直接提问
    # - 必须在指定领域内
    #
    # 输出格式：
    # [
    #   {"question": "问题1", "difficulty": "easy"},
    #   {"question": "问题2", "difficulty": "medium"},
    #   {"question": "问题3", "difficulty": "hard"}
    # ]
    #
    # 特点：领域覆盖广泛，适合生成多样化的 SFT 数据
    """
    def __init__(self):
        self.tag = {
            "Marriage and Relationships": {
                "Dating and Friendship": ["Dating Platforms", "Dating Tips", "Dating Events"],
                "Marriage Management": ["Marital Relationships", "Marriage Law", "Marriage Counseling"],
                "Wedding Planning": ["Wedding Planning", "Wedding Photography", "Wedding Venues"],
                "Relationship Psychology": ["Relationship Psychology", "Communication Skills in Relationships", "Relationship Maintenance"],
                "Emotional Counseling": ["Solving Emotional Issues", "Emotional Repair", "Emotional Growth"],
                "Pre-Marriage Education": ["Pre-Marriage Preparation", "Pre-Marriage Psychology", "Pre-Marriage Legal Knowledge"]
            },
            "Entertainment Gossip": {
                "Celebrity News": ["Celebrity News", "Celebrity Interviews", "Celebrity Charity Events"],
                "Variety Shows": ["Show Recommendations", "Behind the Scenes", "Show Interaction"],
                "Film and TV Reviews": ["Movie Reviews", "TV Series Reviews", "Critics’ Opinions"],
                "Entertainment News": ["Latest Entertainment News", "Entertainment Events", "Exclusive Interviews"],
                "Fan Culture": ["Fan Activities", "Fan Support", "Fan Interactions"],
                "Gossip": ["Celebrity Gossip", "Entertainment Industry Secrets", "Gossip Chasing"]
            },
            "Artificial Intelligence": {
                "Machine Learning": ["Algorithm Principles", "Application Cases", "Learning Resources"],
                "Deep Learning": ["Neural Networks", "Deep Learning Frameworks", "Deep Learning Applications"],
                "Natural Language Processing": ["Language Models", "Text Analysis", "Dialogue Systems"],
                "Computer Vision": ["Image Recognition", "Video Processing", "Vision Algorithms"],
                "Intelligent Robotics": ["Robotics Technology", "Service Robots", "Industrial Robots"],
                "Autonomous Driving": ["Autonomous Driving Technology", "Autonomous Driving Regulations", "Autonomous Driving Testing"]
            },
            "Healthcare": {
                "Disease Prevention and Treatment": ["Common Diseases", "Preventive Measures", "Disease Treatment"],
                "Health and Wellness": ["Dietary Wellness", "Exercise Wellness", "Traditional Chinese Medicine Wellness"],
                "Psychological Counseling": ["Mental Health Issues", "Psychological Therapy", "Psychological Adjustment"],
                "Medical Technology": ["Medical Equipment", "Medical Technology", "Medical Innovation"],
                "Health Insurance": ["Types of Insurance", "Insurance Choices", "Insurance Claims"],
                "Fitness": ["Fitness Methods", "Fitness Equipment", "Fitness Diet"]
            },
            "Pets": {
                "Pet Care": ["Daily Pet Care", "Pet Nutrition", "Pet Behavior"],
                "Pet Medical Care": ["Pet Diseases", "Pet First Aid", "Pet Hospitals"],
                "Pet Training": ["Basic Training", "Behavior Correction", "Training Techniques"],
                "Pet Supplies": ["Toys", "Food", "Care Products"],
                "Pet Adoption": ["Adoption Procedures", "Adoption Conditions", "Adoption Events"],
                "Pet Activities": ["Pet Competitions", "Pet Gatherings", "Pet Festivals"]
            },
            "Environment": {
                "Environmental Protection": ["Ecological Protection", "Pollution Control", "Environmental Monitoring"],
                "Sustainable Development": ["Green Energy", "Circular Economy", "Ecological Agriculture"],
                "Energy Conservation and Emission Reduction": ["Energy-Saving Technology", "Emission Reduction Policies", "Low-Carbon Life"],
                "Waste Sorting": ["Sorting Standards", "Sorting Methods", "Recycling"],
                "Environmental Policies": ["Policy Regulations", "Policy Interpretation", "Policy Impact"],
                "Green Living": ["Green Consumption", "Green Travel", "Green Buildings"]
            },
            "Technology": {
                "Internet": ["Network Technology", "Cybersecurity", "Online Services"],
                "5G Communication": ["5G Technology", "5G Applications", "5G Devices"],
                "Blockchain": ["Blockchain Principles", "Blockchain Applications", "Digital Currency"],
                "Artificial Intelligence": ["AI Technology", "AI Ethics", "AI Industry Applications"],
                "Aerospace": ["Aerospace Technology", "Aircraft", "Space Exploration"],
                "New Energy": ["Solar Energy", "Wind Energy", "New Energy Vehicles", "Energy Storage"]
            },
            "Education and Training": {
                "Preschool Education": ["Choosing Kindergartens", "Early Childhood Education", "Preschool Education Policies"],
                "K12 Education": ["Primary Education", "Secondary Education", "Family Education Guidance"],
                "Higher Education": ["University Major Selection", "Graduate Education", "Higher Education Policies"],
                "Vocational Training": ["Vocational Skills Training", "Professional Certifications", "Career Development Planning"],
                "Online Education": ["Online Course Recommendations", "Distance Education", "Online Learning Tips"],
                "Study Abroad and Immigration": ["Study Abroad Consultation", "Immigration Policies", "Overseas Living Guide"]
            },
            "Career Development": {
                "Career Planning": ["Career Positioning", "Career Development Paths", "Career Transition Guidance"],
                "Job Search Skills": ["Resume Writing", "Interview Skills", "Job Search Channels"],
                "Career Advancement": ["Promotion Strategies", "Workplace Performance", "Leadership Development"],
                "Interpersonal Relationships": ["Colleague Interaction", "Workplace Communication", "Workplace Etiquette"],
                "Entrepreneurship Guidance": ["Entrepreneurship Plans", "Entrepreneurship Resources", "Entrepreneurship Risk Management"],
                "Team Management": ["Team Building", "Team Collaboration", "Team Performance Management"]
            },
            "Finance and Investment": {
                "Stocks": ["Stock Market Analysis", "Stock Investment Strategies", "Stock Research"],
                "Funds": ["Fund Selection", "Systematic Investment Plans", "Fund Risk Management"],
                "Futures": ["Futures Market", "Futures Trading Skills", "Futures Risk Control"],
                "Foreign Exchange": ["Forex Trading", "Forex Market Analysis", "Forex Risk Management"],
                "Insurance": ["Insurance Product Selection", "Insurance Planning", "Insurance Claims"],
                "Financial Planning": ["Personal Finance", "Asset Allocation", "Retirement Planning"]
            },
            "Real Estate and Home Living": {
                "Real Estate Market": ["Market Trends", "Property Price Analysis", "Real Estate Policy Interpretation"],
                "Home Buying Guide": ["Home Selection Tips", "Home Buying Process", "Mortgage Application"],
                "Interior Design": ["Decorating Styles", "Decorating Materials", "Decorating Budget"],
                "Home Living": ["Home Arrangement", "Home Maintenance", "Smart Homes"],
                "Real Estate Policies": ["Policy Updates", "Policy Interpretation", "Policy Impact"],
                "Rental Market": ["Rental Process", "Rental Agreements", "Rental Tips"]
            },
            "Travel and Adventure": {
                "Domestic Travel": ["Destination Recommendations", "Domestic Travel Guides", "Travel Safety"],
                "International Travel": ["Visa Applications", "International Travel Guides", "Cultural Adaptation"],
                "Outdoor Adventures": ["Hiking", "Mountain Climbing", "Wilderness Survival Skills"],
                "Travel Guides": ["Travel Planning", "Travel Budget", "Travel Packing Lists"],
                "Travel Equipment": ["Backpack Selection", "Outdoor Gear", "Travel Essentials"],
                "Travel Photography": ["Photography Tips", "Travel Photography Works", "Photography Equipment Recommendations"]
            },
            "Food and Cooking": {
                "Food Recommendations": ["Local Delicacies", "Food Rankings", "Restaurant Recommendations"],
                "Cooking Skills": ["Basic Cooking", "Creative Cooking", "Cooking Tool Usage"],
                "Ingredient Selection": ["Ingredient Selection Tips", "Seasonal Ingredients", "Organic Ingredients"],
                "Food Culture": ["Food Culture", "Local Food Customs", "Dietary Health"],
                "Healthy Eating": ["Balanced Nutrition", "Healthy Recipes", "Dietary Wellness"],
                "Baking and Desserts": ["Dessert Making", "Baking Skills", "Dessert Ingredients"]
            },
            "Culture and Arts": {
                "Literature": ["Literary Works", "Literary Criticism", "Creative Writing Skills"],
                "Music": ["Music Styles", "Music Production", "Music Appreciation"],
                "Painting": ["Painting Techniques", "Painting Schools", "Painting Appreciation"],
                "Sculpture": ["Sculpture Art", "Sculpture Creation", "Sculpture Materials"],
                "Theater": ["Theater Performance", "Theater Creation", "Theater History"],
                "Film": ["Film Recommendations", "Film Reviews", "Film Production"]
            },
            "Sports and Fitness": {
                "Sports Events": ["Event Broadcasts", "Event Analysis", "Event History"],
                "Fitness Methods": ["Fitness Tutorials", "Fitness Plans", "Fitness Diet"],
                "Sports Equipment": ["Equipment Recommendations", "Equipment Usage", "Equipment Maintenance"],
                "Sports Celebrities": ["Celebrity Introductions", "Celebrity Interviews", "Celebrity Events"],
                "Sports Policies": ["Policy Interpretation", "Policy Impact", "Policy Updates"],
                "Sports Industry": ["Industry Trends", "Industry Investment", "Industry Cases"]
            },
            "Military and National Defense": {
                "Military News": ["News Reports", "News Analysis", "Military Updates"],
                "Defense Technology": ["Technology Advancements", "Technology Applications", "Innovative Technologies"],
                "Weapons and Equipment": ["Equipment Introduction", "Equipment Comparison", "Equipment Maintenance"],
                "Military History": ["Historical Events", "Historical Battles", "Historical Figures"],
                "Military Service System": ["Service Regulations", "Enlistment Process", "Veterans' Policies"],
                "National Security": ["Security Policies", "Security Education", "Security Awareness"]
            },
            "Social Welfare": {
                "Charity Donations": ["Donation Channels", "Donation Impact", "Donation Stories"],
                "Volunteer Services": ["Service Projects", "Service Training", "Volunteer Stories"],
                "Public Welfare Activities": ["Activity Organization", "Activity Participation", "Activity Impact"],
                "Public Welfare Organizations": ["Organization Introductions", "Organization Activities", "Organization Cooperation"],
                "Social Assistance": ["Assistance Targets", "Assistance Methods", "Assistance Policies"],
                "Spreading Love": ["Spreading Methods", "Spreading Activities", "Spreading Impact"]
            },
            "Automotive and Transportation": {
                "Automotive News": ["New Car Releases", "Car Reviews", "Automotive Trends"],
                "Driving Skills": ["Safe Driving", "Fuel-Efficient Driving", "Driver Training"],
                "Vehicle Maintenance": ["Routine Maintenance", "Fault Diagnosis", "Repair Services"],
                "Traffic Laws": ["Law Interpretation", "Safety Education", "Law Updates"],
                "New Energy Vehicles": ["Technical Features", "Market Dynamics", "Policy Support"],
                "Smart Transportation": ["Technology Applications", "Smart Systems", "Future Trends"]
            },
            "E-commerce": {
                "Online Shopping": ["Shopping Guides", "User Reviews", "Promotions"],
                "E-commerce Operations": ["Operations Management", "Market Analysis", "Customer Service"],
                "Cross-border E-commerce": ["International Logistics", "Tariff Policies", "Market Analysis"],
                "E-commerce Policies": ["Policy Interpretation", "Policy Impact", "Compliance Operations"],
                "E-commerce Marketing": ["Marketing Strategies", "Advertising Placement", "User Analysis"],
                "E-commerce Logistics": ["Logistics Delivery", "Inventory Management", "Logistics Technology"]
            },
            "Gaming and Animation": {
                "Online Games": ["Popular Games", "Game Reviews", "Gaming Communities"],
                "Single-player Games": ["Classic Games", "Game Guides", "Game Recommendations"],
                "Animation Works": ["Popular Anime", "Anime Characters", "Anime Production"],
                "Game Guides": ["Guide Sharing", "Skill Exchange", "Guide Videos"],
                "Animation Industry": ["Industry Trends", "Market Analysis", "Industry Policies"],
                "Game Merchandise": ["Merchandise Products", "Collecting Guides", "Merchandise Events"]
            },
            "Infant and Child Education": {
                "Early Education": ["Educational Philosophy", "Educational Methods", "Educational Toys"],
                "Maternal and Infant Care": ["Care Knowledge", "Care Skills", "Care Products"],
                "Child Psychology": ["Psychological Development", "Emotion Management", "Psychological Counseling"],
                "Parent-child Relationship": ["Parent-child Activities", "Parent-child Communication", "Parent-child Education"],
                "Baby Products": ["Product Selection", "Safety Standards", "Product Recommendations"],
                "Child Health": ["Healthy Growth", "Nutritional Diet", "Disease Prevention"]
            },
            "Senior Life": {
                "Elderly Care Policies": ["Policy Interpretation", "Policy Consultation", "Policy Implementation"],
                "Senior Health": ["Health Checkups", "Disease Prevention", "Healthy Eating"],
                "Senior Activities": ["Cultural Activities", "Sports Activities", "Social Activities"],
                "Senior Psychology": ["Psychological Adjustment", "Psychological Health", "Psychological Support"],
                "Elderly Care Institutions": ["Institution Selection", "Service Quality", "Institution Evaluation"],
                "Senior Products": ["Assistance Products", "Health Products", "Living Products"]
            },
            "Psychological Counseling": {
                "Mental Health": ["Mental Maintenance", "Mental Problem Prevention", "Mental Health Education"],
                "Psychological Disorders": ["Disorder Identification", "Disorder Treatment", "Disorder Management"],
                "Counseling Skills": ["Counseling Methods", "Communication Skills", "Case Studies"],
                "Psychological Tests": ["Test Types", "Test Applications", "Test Interpretation"],
                "Psychological Research": ["Research Trends", "Research Methods", "Research Results"],
                "Psychological Guidance": ["Guidance Strategies", "Guidance Cases", "Guidance Resources"]
            },
            "Religion and Belief": {
                "Religious Culture": ["Cultural Traditions", "Cultural Festivals", "Cultural Influence"],
                "Religious History": ["Historical Development", "Key Events", "Historical Figures"],
                "Religious Art": ["Art Forms", "Art Works", "Art Value"],
                "Religious Policies": ["Policy Regulations", "Policy Interpretation", "Policy Impact"],
                "Religious Activities": ["Activity Organization", "Activity Participation", "Activity Significance"],
                "Faith Discussions": ["Meaning of Faith", "Faith Conflicts", "Faith Diversity"]
            },
            "Agriculture and Rural Development": {
                "Agricultural Technology": ["Technology Applications", "Technological Innovation", "Technology Promotion"],
                "Rural Development": ["Development Planning", "Development Models", "Development Cases"],
                "Farmer Life": ["Life Improvement", "Quality of Life", "Living Customs"],
                "Agricultural Products Market": ["Market Analysis", "Market Trends", "Market Transactions"],
                "Agricultural Policies": ["Policy Support", "Policy Interpretation", "Policy Implementation"],
                "Rural Tourism": ["Tourism Development", "Tourism Projects", "Tourism Experience"]
            },
            "Urban Planning": {
                "Urban Planning": ["Planning Philosophy", "Planning Methods", "Planning Cases"],
                "Urban Design": ["Design Philosophy", "Design Elements", "Design Practice"],
                "Infrastructure Development": ["Development Planning", "Development Management", "Development Technology"],
                "Urban Transportation": ["Transportation Planning", "Transportation Management", "Transportation Tools"],
                "Urban Greening": ["Greening Layout", "Greening Technology", "Greening Effects"],
                "Protection of Historic Cities": ["Protection Policies", "Protection Measures", "Protection Cases"]
            },
            "Laws and Regulations": {
                "Civil Law": ["General Principles", "Property Law", "Contract Law"],
                "Criminal Law": ["General Principles", "Types of Crimes", "Punishment Systems"],
                "Administrative Law": ["Administrative Regulations", "Administrative Litigation", "Administrative Reconsideration"],
                "Economic Law": ["Corporate Law", "Tax Law", "Intellectual Property Law"],
                "International Law": ["Public International Law", "Private International Law", "International Trade Law"],
                "Legal Consultation": ["Consultation Services", "Legal Aid", "Legal Education"]
            },
            "Art": {
                "Painting": ["Painting Techniques", "Painting Styles", "Painting Works"],
                "Sculpture": ["Sculpture Materials", "Sculpture Styles", "Sculpture Creation"],
                "Design": ["Design Philosophy", "Design Methods", "Design Works"],
                "Photography": ["Photography Techniques", "Photography Themes", "Photography Works"],
                "Calligraphy": ["Calligraphy Art", "Calligraphy Styles", "Calligraphy Works"],
                "Handicrafts": ["Craft Making", "Craft Materials", "Craft Culture"]
            },
            "Marketing": {
                "Market Research": ["Research Methods", "Research Tools", "Research Reports"],
                "Marketing Strategies": ["Strategy Formulation", "Strategy Execution", "Strategy Evaluation"],
                "Brand Management": ["Brand Positioning", "Brand Promotion", "Brand Maintenance"],
                "Advertising": ["Creative Advertising", "Advertising Media", "Advertising Effectiveness"],
                "Public Relations": ["Event Planning", "Event Execution", "Event Evaluation"],
                "Channel Development": ["Channel Expansion", "Channel Management", "Channel Optimization"]
            },
            "Astronomy and Geography": {
                "Astronomy": ["Astronomical Observations", "Astronomical Phenomena", "Astronomical Research"],
                "Geography": ["Geographical Knowledge", "Geographical Exploration", "Geographical Education"],
                "Geology": ["Geological Structure", "Geological Survey", "Geological Protection"],
                "Meteorology": ["Weather Forecasting", "Weather Disasters", "Weather Services"],
                "Space Exploration": ["Space Exploration", "Interstellar Travel", "Extraterrestrial Life"],
                "Geographical Information Systems": ["GIS Technology", "GIS Applications", "GIS Development"]
            },
            "Education and Exams": {
                "College Entrance Exam Coaching": ["Preparation Strategies", "Practice Tests", "Exam Policy Interpretation"],
                "Graduate School Entrance Exam Coaching": ["Preparation Planning", "Specialty Coaching", "Psychological Adjustment"],
                "Civil Service Exams": ["Exam Techniques", "Essay Writing Guidance", "Interview Preparation"],
                "Teaching Qualification Exams": ["Exam Process", "Interview Skills", "Teaching Ability Improvement"],
                "Foreign Language Exams": ["CET-4/CET-6", "IELTS/TOEFL", "Foreign Language Speaking Training"],
                "Professional Qualification Exams": ["Exam Subjects", "Career Development", "Qualification Certification"]
            },
            "Cybersecurity": {
                "Cybersecurity Protection": ["Protection Measures", "Security Tools", "Protection Strategies"],
                "Hacker Attack and Defense": ["Attack and Defense Drills", "Security Vulnerabilities", "Hacking Techniques"],
                "Data Encryption": ["Encryption Technology", "Data Protection", "Encryption Strategies"],
                "Information Leak Prevention": ["Leakage Risks", "Prevention Measures", "Emergency Response"],
                "Cybersecurity Policies": ["Policy Interpretation", "Regulations and Standards", "Policy Updates"],
                "Cybersecurity Incidents": ["Incident Analysis", "Incident Tracking", "Incident Prevention"]
            },
            "Fashion and Trends": {
                "Clothing Matching": ["Everyday Outfits", "Dressing for Occasions", "Fashion Trends"],
                "Beauty and Skincare": ["Skincare Knowledge", "Makeup Skills", "Beauty Products"],
                "Fashion Accessories": ["Jewelry Matching", "Accessory Selection", "Trendy Accessories"],
                "Trend Analysis": ["Fashion Week", "Trend Analysis", "Trend Forecasting"],
                "Fashion Bloggers": ["Blogger Recommendations", "Blogger Styles", "Blogger Influence"],
                "Fashion Brands": ["Brand Stories", "Brand Series", "Brand Events"]
            },
            "Mental Health": {
                "Emotion Management": ["Emotion Recognition", "Emotion Regulation", "Emotion Expression"],
                "Stress Management": ["Stress Sources", "Stress Relief Techniques", "Stress Management"],
                "Interpersonal Relationships": ["Communication Skills", "Conflict Resolution", "Social Skills"],
                "Self-Awareness": ["Self-Exploration", "Self-Evaluation", "Personal Growth"],
                "Psychological Adjustment": ["Adjustment Methods", "Psychological Balance", "Psychological Resilience"],
                "Psychological Disorder Prevention": ["Disorder Knowledge", "Prevention Measures", "Health Promotion"]
            },
            "Agricultural Technology": {
                "Smart Agriculture": ["Smart Technology", "Precision Agriculture", "Agricultural Big Data"],
                "Agricultural Mechanization": ["Mechanization Applications", "Technological Innovation", "Mechanization Maintenance"],
                "Agricultural Product Processing": ["Processing Technology", "Product Innovation", "Quality Control"],
                "Agricultural Innovation": ["Innovation Cases", "Innovation Policies", "Innovation-Driven Development"],
                "Agricultural Policies": ["Policy Support", "Policy Interpretation", "Policy Implementation"],
                "Agricultural Market Analysis": ["Market Trends", "Demand Analysis", "Price Fluctuations"]
            },
            "Digital Products": {
                "Smartphone Reviews": ["Performance Testing", "User Experience", "New Releases"],
                "Computer Hardware": ["Hardware Configuration", "Hardware Upgrades", "Hardware Maintenance"],
                "Digital Cameras": ["Camera Selection", "Photography Tips", "Camera Maintenance"],
                "Wearable Devices": ["Device Functions", "Health Monitoring", "Smart Interactions"],
                "Routers": ["Router Setup", "Signal Optimization", "Network Security"],
                "Digital Accessories": ["Accessory Selection", "Device Protection", "Accessory Recommendations"]
            },
            "Home Decoration": {
                "Decoration Styles": ["Modern Minimalism", "Classical Chinese Style", "Luxurious European Style"],
                "Decoration Materials": ["Material Selection", "Material Environmental Protection", "Material Costs"],
                "Interior Design": ["Space Planning", "Furniture Selection", "Color Matching"],
                "Soft Decoration": ["Curtain Selection", "Bedding Matching", "Decorative Paintings"],
                "Feng Shui": ["Feng Shui Layout", "Feng Shui Taboos", "Feng Shui Improvements"],
                "Renovation Construction": ["Construction Process", "Construction Supervision", "Construction Safety"]
            },
            "History and Culture": {
                "Chinese History": ["Ancient History", "Modern History", "History Education"],
                "World History": ["Origins of Civilization", "Historical Events", "International Relations"],
                "Archaeological Discoveries": ["Site Excavation", "Cultural Relic Protection", "Archaeological Techniques"],
                "Historical Figures": ["Biographies", "Character Evaluations", "Historical Impact"],
                "Cultural Heritage": ["Heritage Protection", "Heritage Value", "Heritage Inheritance"],
                "Historical Research": ["Research Methods", "Academic Achievements", "Research Trends"]
            },
            "Travel Guides": {
                "Independent Travel Guides": ["Destination Recommendations", "Itinerary Planning", "Accommodation Selection"],
                "Group Travel Guides": ["Tour Agency Selection", "Group Activities", "Group Travel Advantages"],
                "Tourism Route Planning": ["Route Design", "Special Routes", "Theme Travel"],
                "Money-Saving Travel Tips": ["Budget Planning", "Spending Guides", "Discount Information"],
                "Travel Safety": ["Safety Tips", "Emergency Handling", "Insurance Selection"],
                "Travel Visas": ["Visa Applications", "Visa Policies", "Visa Documentation"]
            },
            "Food Sharing": {
                "Recipe Sharing": ["Recipe Sharing", "Cooking Skills", "Ingredient Selection"],
                "Food Recommendations": ["Special Dishes", "Local Snacks", "Restaurant Recommendations"],
                "Food Exploration": ["Exploration Guides", "Shop Reviews", "Food Maps"],
                "Food Photography": ["Photography Skills", "Food Presentation", "Visual Display"],
                "Food Reviews": ["Dish Reviews", "Restaurant Reviews", "Ingredient Reviews"],
                "Food Competitions": ["Competition Information", "Participation Guidelines", "Award-Winning Works"]
            },
            "Film and Entertainment": {
                "Movie Recommendations": ["New Movie Alerts", "Classic Movies", "Movie Rankings"],
                "TV Series Reviews": ["Popular Drama Reviews", "Series Recommendations", "Plot Analysis"],
                "Variety Show Reviews": ["Program Highlights", "Guest Performances", "Program Creativity"],
                "Online Series": ["Popular Online Series", "Online Series Production", "Online Series Trends"],
                "Short Videos": ["Short Video Creation", "Short Video Platforms", "Short Video Marketing"],
                "Film Production": ["Production Process", "Behind the Scenes", "Production Techniques"]
            },
            "Sports Activities": {
                "Ball Sports": ["Football", "Basketball", "Volleyball"],
                "Track and Field": ["Running", "Long Jump", "Throwing"],
                "Water Sports": ["Swimming", "Rowing", "Surfing"],
                "Winter Sports": ["Skiing", "Ice Skating", "Sledding"],
                "Extreme Sports": ["Rock Climbing", "Skydiving", "Extreme Cycling"],
                "Sports Events": ["International Events", "Domestic Events", "Local Events"]
            },
            "Entrepreneurship and Investment": {
                "Entrepreneurship Guidance": ["Entrepreneurship Plans", "Market Analysis", "Entrepreneurship Mindset"],
                "Investment and Finance": ["Investment Strategies", "Asset Management", "Risk Control"],
                "Entrepreneurship Policies": ["Policy Interpretation", "Policy Support", "Policy Utilization"],
                "Entrepreneurship Cases": ["Success Stories", "Lessons Learned", "Case Analysis"],
                "Venture Capital": ["Investment Opportunities", "Investment Evaluation", "Investment Negotiation"],
                "Entrepreneurship Financing": ["Financing Channels", "Financing Strategies", "Financing Agreements"]
            },
            "Music and Dance": {
                "Music Appreciation": ["Music Styles", "Music Works", "Musicians"],
                "Instrumental Performance": ["Instrument Selection", "Performance Techniques", "Instrument Maintenance"],
                "Dance Performance": ["Dance Types", "Performance Techniques", "Performance Opportunities"],
                "Music Production": ["Music Creation", "Music Recording", "Music Publishing"],
                "Music Education": ["Education Methods", "Educational Resources", "Education Policies"],
                "Dance Choreography": ["Choreography Techniques", "Choreography Creativity", "Choreography Practice"]
            },
            "National Defense and Military": {
                "Military Strategy": ["Strategy Analysis", "Strategy Planning", "Strategy Implementation"],
                "Military Training": ["Basic Training", "Tactical Training", "Special Forces Training"],
                "Weapons Development": ["Equipment Introduction", "Research and Development Updates", "Technological Innovation"],
                "Military History": ["Historical Battles", "Historical Figures", "Historical Events"],
                "National Defense Education": ["Educational Content", "Educational Methods", "Educational Significance"],
                "Military Exercises": ["Exercise Types", "Exercise Scale", "Exercise Objectives"]
            }
        }
        
        # 任务类型（增强场景多样性，参考论文中的常见交互场景）
        self.task_types = [
            "Daily Conversation",
            "Creative Task",
            "Role Playing",
            "Problem Solving",
            "Educational Explanation",
            "Emotional Support",
            "Information Retrieval"
        ]
    
    def build_prompt(self, theme, domain):
        """
        Generates the formatted prompt for LLM input based on the theme and domain.

        Parameters:
        theme (str): The main theme of the questions.
        domain (str): The domain under the given theme.

        Returns:
        str: The formatted prompt for generating questions.
        """
        prompt = f"""
Now we need to create high-quality SFT data for LLM training, so we need you to produce a batch of such data. You only
need to create Questions. I will give you a theme for SFT data Questions. You need to create three
Questions of different difficulty levels based on this new theme.\\
Your Questions must meet the following requirements:\\
1. You must strictly create only three Questions at a time. These three Questions must be in the domain of {domain}
and the Questions should align with the given theme of {theme}.\\
2. The Questions you create must have context and sufficient information; they should not be abrupt and directly ask the
question.\\
3. Your reply must strictly follow the format below. Your Questions need to be included between [Question Start] and
[Question End], and the difficulty level should be indicated at the beginning, as in the following format:\\

[Easy][Question Start]Question[Question End]

[Medium][Question Start]Question[Question End]

[Hard][Question Start]Question[Question End]

4. Your Questions of different difficulty levels should be distinct and actually reflect the different levels of difficulty.\\
\quad \\

Now it's your turn. Please provide the three Questions of different difficulty levels you created about the theme of {theme} for {domain}, according to the requirements.
"""
        return prompt

    
@PROMPT_REGISTRY.register()
class CondorCritiquePrompt(PromptABC):
    """
    # 用途：对模型的回答进行批评，指出优缺点
    # 应用场景：在 CondorRefiner 算子中使用（第一步）
    #
    # 输入：
    # - question：用户问题
    # - answer：模型回答
    #
    # 输出格式：
    # [Critique Start]
    # [Strength Start]优点[Strength End]
    # [Weakness Start]缺点[Weakness End]
    # [Suggestion Start]改进建议[Suggestion End]
    # [Critique End]
    #
    # 用途：为后续的回答改进提供反馈
    """
    def __init__(self):
        pass
    
    def build_prompt(self, question, answer):
        dialogue = [question, answer]
        base_critique_prompt = f"""
There is now a user’s question and a model’s response. You need to write a critique for this response, pointing out the
strengths and weaknesses of the model’s answer to help the model improve its response.

Your critique must strictly adhere to the following format:

[Critique Start]

[Strength Start]Strength[Strength End]

[Weakness Start]Weakness[Weakness End]

[Suggestion Start]Suggestion[Suggestion End]

[Critique End]

Here is the user’s question and the model’s response: {dialogue}

Now it’s your turn. Please provide your Critique as required:
        """
        return base_critique_prompt

@PROMPT_REGISTRY.register()
class CondorRefinePrompt(PromptABC):
    """
    # 用途：基于批评反馈改进模型的回答
    # 应用场景：在 CondorRefiner 算子中使用（第二步）
    #
    # 输入：
    # - question：用户问题
    # - answer：原始回答
    # - critique：批评反馈
    #
    # 输出格式：
    # [Improved Answer Start]改进后的回答[Improved Answer End]
    #
    # 工作流程：
    # CondorQuestionPrompt → 生成问题
    # → 模型回答
    # → CondorCritiquePrompt → 批评
    # → CondorRefinePrompt → 改进回答
    #
    # 特点：通过批评-改进循环提升回答质量
    """
    def __init__(self):
        pass

    def build_prompt(self, question, answer, critique):
        base_refine_prompt = """
Now there is a user's question, a model's answer, and the user's feedback. Please help modify the model's answer based on the user's feedback to make it better.
Your improved answer must strictly adhere to the following format:

[Improved Answer Start]Your answer[Improved Answer End]

Below is the user's question, the model's answer, and the feedback:
[Question Start]{question}[Question End]
[Answer Start]{answer}[Answer End]
[Feedback Start]{critique}[Feedback End]

Now it's your turn, please provide your improved answer as required:
        """
        return base_refine_prompt.format(question=question, answer=answer, critique=critique)

@PROMPT_REGISTRY.register()
class LanguageFilterPrompt(PromptABC):
    """
    # 用途：识别文本的语言类型
    # 应用场景：在 LanguageFilter 算子中使用，过滤特定语言的数据
    #
    # 支持的语言：
    # - English
    # - Chinese (Simplified/Traditional)
    # - Spanish
    # - French
    # - German
    # - Japanese
    # - Korean
    # - Russian
    # - Arabic
    # - Portuguese
    # - Italian
    # - Dutch
    # - Polish
    # - Turkish
    # - Vietnamese
    # - Thai
    # - Indonesian
    # - Hindi
    # - Other
    #
    # 输出格式：
    # Language: English
    #
    # 特点：简单直接的语言识别，用于数据清洗和分类
    """
    def __init__(self):
        pass
    
    def build_prompt(self, text):
        prompt='''You are a language identification expert. Your task is to identify the language of the given text input.

        Follow these rules:You are a language identification expert. Your task is to identify the language of the given text input.

    - Respond with the ISO 639-1 two-letter language code (e.g., "en", "fr", "zh", "ar").
        - If the text contains multiple languages, identify the dominant one.
        - If the language is not identifiable, respond with "Unknown".
        - Do not translate or explain. Output only the language name.

        Here are some examples:

        Example 1:
        Text: "Hello, how are you?"
        Language: en

        Example 2:
        Text: "Je suis très heureux de vous rencontrer."
        Language: fr

        Example 3:
        Text: "これは日本語の文です。"
        Language: ja

        Example 4:
        Text: "¿Dónde está la estación de tren?"
        Language: es

        Example 5:
        Text: "مرحبا، كيف حالك؟"
        Language: ar

        Example 6:
        Text: "Guten Morgen! Wie geht's dir?"
        Language: de

        Example 7:
        Text: "你好，我是一个程序员。"
        Language: zh

        Example 8:
        Text: "Привет, как дела?"
        Language: ru

        Now, identify the language of the following text:

        Text: "{text}"
        Language:
        '''
        return prompt.format(text=text)



@PROMPT_REGISTRY.register()
class SFTFromScratchGeneratorPrompt(PromptABC):
    """
    Prompt for generating brand-new SFT samples from scratch.
    # 用途：不依赖任何种子数据，从零生成高质量 SFT 训练样本
    # 应用场景：在 SFTFromScratchGenerator 算子中使用
    #
    # 核心原则：
    # 1. 结构优秀：instruction、input、output、domain 四个字段
    # 2. 质量标准：准确、自然、完整、深度适当
    # 3. 多样性：不同难度、不同场景、不同用户角色
    # 4. 安全伦理：无害、无歧视、中立、平衡
    # 5. 技术格式：单行 JSON，正确转义
    #
    # 输入：
    # - domain_keys：可选的领域列表（如 "coding, math, writing"）
    #
    # 输出格式：
    # {"instruction": "...", "input": "...", "output": "...", "domain": "..."}
    #
    # 示例：
    # {
    #   "instruction": "Create a Python function that calculates compound interest",
    #   "input": "",
    #   "output": "def compound_interest(principal, rate, time, n=1):...",
    #   "domain": "coding"
    # }
    #
    # 特点：
    # - 完全自主生成，不需要种子数据
    # - 强调多样性和真实性
    # - 适合大规模生成训练数据
    """
    def __init__(self):
        pass

    def build_prompt(self, domain_keys: str) -> str:
        system_prompt = """You are a sophisticated data generation assistant specialized in creating high-quality Supervised Fine-Tuning (SFT) datasets for large language models.

Your mission is to generate diverse, realistic, and instruction-following training samples that will help models become more helpful, accurate, and aligned with human preferences.

## Core Principles:

**1. Structural Excellence:**
- instruction: Clear, specific, and actionable user request
- input: Contextual information when relevant (empty string if none needed)
- output: Comprehensive, accurate, and genuinely helpful response
- domain: Single domain classification from the provided taxonomy

**2. Quality Standards:**
- Responses must be factually accurate and demonstrate expertise
- Use natural, conversational language appropriate to the context
- Provide complete solutions that fully address the instruction
- Maintain consistency between instruction complexity and response depth
- Include relevant examples, explanations, or step-by-step guidance when beneficial

**3. Diversity Requirements:**
- Vary instruction phrasing and complexity levels
- Mix different user personas and contexts
- Include both simple and complex scenarios within each domain
- Generate instructions that reflect real-world use cases

**4. Safety & Ethics:**
- No harmful, illegal, discriminatory, or misleading content
- Respect privacy and avoid generating personal information
- Maintain neutrality on controversial topics
- Provide balanced perspectives when appropriate

**5. Technical Format:**
- Output valid JSON in a single line with no formatting
- Properly escape special characters in strings
- Ensure all required fields are present and correctly typed"""
        
        user_prompt = f"""Generate ONE premium-quality SFT training sample as a single-line JSON object.

## Requirements:
- **instruction**: A realistic user request that varies in style, complexity, and specificity
- **input**: Additional context when it enhances the scenario (otherwise empty string)
- **output**: A comprehensive, expert-level response that fully satisfies the instruction
- **domain**: Select the most appropriate domain from: {domain_keys}

## Quality Checklist:
✓ Instruction is clear and represents authentic user needs
✓ Response demonstrates expertise and provides genuine value
✓ Appropriate level of detail for the complexity of the request
✓ Natural, human-like language throughout
✓ Perfect JSON formatting in a single line

## Diversity Goals:
- Mix formal/informal language styles
- Include various difficulty levels and user contexts
- Represent different cultural perspectives when relevant
- Balance theoretical knowledge with practical applications

## Format Example:
{{"instruction": "Create a Python function that calculates compound interest with error handling", "input": "", "output": "def compound_interest(principal, rate, time, n=1):\\n    if principal <= 0 or rate < 0 or time < 0 or n <= 0:\\n        raise ValueError('Invalid input: principal must be positive, rate and time non-negative, n positive')\\n    return principal * (1 + rate/n)**(n*time)\\n\\n# Example usage:\\n# result = compound_interest(1000, 0.05, 2, 4)", "domain": "coding"}}

Output only the JSON - no explanations or additional text."""
        return system_prompt + "\n\n" + user_prompt


@PROMPT_REGISTRY.register()
class TextQuestionGeneratorPrompt(PromptABC):
    """
    # 用途：从文本中生成多个高质量问题
    # 应用场景：在 TextQuestionGenerator 算子中使用
    #
    # 核心功能：
    # 1. 分析文本内容，提取关键信息
    # 2. 生成指定数量的问题
    # 3. 输出 JSON 数组格式
    #
    # 参数：
    # - number：生成问题的数量
    # - custom_prompt：用户自定义额外要求（可选）
    #
    # 特点：
    # - 问题必须基于原文，不能包含外部信息
    # - 覆盖文本的多个主题和角度
    # - 问题表述自然，不使用"文章中""报告中"等元信息
    """

    def __init__(self, number: int = 5, custom_prompt: str = ""):
        self.number = number
        self.custom_prompt = custom_prompt

    def build_prompt(self, text: str) -> str:
        """
        构建问题生成的提示词

        Args:
            text: 待分析的文本内容

        Returns:
            格式化后的提示词
        """
        text_length = len(text)

        # 处理自定义提示词
        custom_section = ""
        if self.custom_prompt:
            custom_section = f"\n6. Additional requirement: {self.custom_prompt}"

        base_prompt = f"""# Role: Text Question Generation Expert
## Profile:
- Description: You are an expert in text analysis and question design, capable of extracting key information from complex passages and producing high-quality questions for fine-tuning datasets.
- Input Length: {text_length} characters
- Output Goal: Generate at least {self.number} high-quality questions suitable for training data.

## Skills:
1. Comprehend the source text thoroughly and identify core concepts, facts, and logical structures.
2. Design questions with clear answer orientation that cover multiple aspects of the text.
3. Balance difficulty and variety to ensure representative coverage of the content.
4. Enforce strict formatting so the output can be consumed programmatically.

## Workflow:
1. **Text Parsing**: Read the entire passage, segment it, and capture key entities, events, metrics, and conclusions.
2. **Question Design**: Select the most informative focal points to craft questions.
3. **Quality Check**: Validate each question to ensure:
   - The answer can be located directly in the original text.
   - Questions do not duplicate topics or angles.
   - Wording is precise, unambiguous, and uses natural interrogative phrasing.

## Constraints:
1. Every question must be grounded strictly in the provided text; no external information or hypothetical scenarios.
2. Cover diverse themes, layers, or perspectives from the passage; avoid clustering around one segment.
3. Do not include questions about meta information (author, chapters, table of contents, etc.).
4. Avoid phrases such as "in the report/article/literature/table"; questions must read naturally.
5. Produce at least {self.number} questions with consistent formatting.
6. **CRITICAL: The generated questions MUST be in the SAME LANGUAGE as the input text.** If the input is in Chinese, generate Chinese questions. If the input is in English, generate English questions. Do not translate or switch languages.{custom_section}

## Output Format:
- Return a valid JSON array containing only strings.
- Use double quotes for all strings.
- Follow this exact structure:
["Question 1", "Question 2", "..."]

## Output Example:
["What core elements should an AI ethics framework include?", "What new regulations does the Civil Code have for personal data protection?"]

## Text to Analyze:
{text}"""

        return base_prompt


@PROMPT_REGISTRY.register()
class TextAnswerGeneratorPrompt(PromptABC):
    """
    # 用途：基于问题和上下文生成准确的答案
    # 应用场景：在 TextAnswerGenerator 算子中使用
    #
    # 核心功能：
    # 1. 分析给定的参考内容
    # 2. 针对问题提取关键信息
    # 3. 生成准确、详细的答案
    #
    # 参数：
    # - custom_prompt：用户自定义额外要求（可选）
    # - output_format：输出格式说明（可选）
    #
    # 特点：
    # - 答案必须基于给定内容，不能编造
    # - 答案准确且与问题相关
    # - 答案全面详细，适合用于微调大语言模型
    """

    def __init__(self, custom_prompt: str = "", output_format: str = ""):
        self.custom_prompt = custom_prompt
        self.output_format = output_format

    def build_prompt(self, text: str, question: str) -> str:
        """
        构建答案生成的提示词

        Args:
            text: 参考内容/上下文
            question: 需要回答的问题

        Returns:
            格式化后的提示词
        """
        # 处理自定义约束（追加为独立的补充要求段，避免与编号约束混排）
        custom_section = ""
        if self.custom_prompt:
            custom_section = f"\n\n## 补充要求：\n{self.custom_prompt}"

        # 处理输出格式
        output_format_section = ""
        if self.output_format:
            output_format_section = f"\n\n## 输出格式：\n{self.output_format}"

        base_prompt = f"""# 角色：IT 运维 QA 数据集专家（SRE/DBA）

## 简介：
你是一名专注于生成 IT 运维（ITIL/DevOps）微调数据集的专家，擅长从给定参考内容中提炼准确、实用、高相关性的答案，确保答案对故障排查、事件响应、系统管理或架构设计具有实质参考价值。

## 能力要求：
1. 答案必须严格基于给定内容，不得编造。
2. 答案必须准确，与问题直接相关，提供明确的解决方案或解释。
3. 答案逻辑清晰，结构化程度适合 SRE/DBA 快速阅读和使用。
4. 优先提取文本中的精确路径、命令、指标和现象等可操作信息。

## 工作流程：
1. 以运维人员视角仔细分析给定的参考内容。
2. 从内容中提炼关键运维信息（命令、配置项、症状现象、根因线索）。
3. 针对问题生成准确、可落地的答案。
4. 确认答案的准确性、相关性与运维实用价值后输出。

## 参考内容：

------ 参考内容开始 ------
{text}
------ 参考内容结束 ------

## 问题：
{question}

## 约束条件：
1. 答案必须基于给定的参考内容，不得引入参考内容之外的信息。
2. 答案必须准确，不允许出现任何编造内容。
3. 答案须全面详细，包含所有必要信息，适合用于大语言模型微调训练。
4. **语言一致性（硬性约束）**：答案必须与参考内容和问题使用相同的语言。若内容为中文，则用中文作答；若为英文，则用英文作答，禁止自行翻译或切换语言。
5. **禁止元信息开篇（硬性约束）**：答案中不得出现「根据参考内容」「参考内容明确指出」「参考内容显示」「原文指出」「原文中提到」「根据原文」「根据文档」「根据提供的信息」「在步骤X」「根据流程图」「根据表格」等元信息引用短语，必须直接陈述事实。例如：不得写「根据参考文档指出，A 是 B」，应直接写「A 是 B」。
6. **禁止反向表述**：对于枚举型问题（如「需要填哪些参数 / 包含哪些字段 / 记录哪些信息」），须直接列举（「需要填写的参数包括：XX、YY、ZZ」），不得使用反向表述，如「应检查并重新确认 XX 是否正确」「应关注 XX」「需要注意 XX」。
7. **保持运维实用价值**：对于故障排查、配置说明或最佳实践类问题，优先提取可操作步骤、精确参数和明确诊断标准，避免泛泛的理论描述。{custom_section}{output_format_section}

## 答案："""

        return base_prompt


@PROMPT_REGISTRY.register()
class QuestionCritiquePrompt(PromptABC):
    """
    # 用途：对生成的问题进行批评，指出问题的优缺点
    # 应用场景：在 QuestionRefiner 算子中使用（第一步）
    #
    # 批评维度（8个）：
    # 1. 元信息检查：问题是否包含章节、步骤、行号、流程图等文档结构引用
    # 2. 具体性平衡：问题是否过于细节或过于宽泛
    # 3. 运维视角：问题是否符合运维人员的提问习惯
    # 4. 表述清晰度：问题是否无歧义，上下文完整
    # 5. 实用价值：问题对实际排障是否有帮助
    # 6. 原文支撑：问题是否能在原文中找到对应内容
    # 7. 回答边界与可验证目标
    # 8. 答案泄漏检查
    #
    # 输入：
    # - context：参考上下文
    # - question：生成的问题
    #
    # 输出格式：
    # [Critique Start]
    # [各维度检查结果]
    # [Overall Assessment]
    # [Suggestion]
    # [Critique End]
    """
    def __init__(self):
        pass

    def build_prompt(self, context: str, question: str) -> str:
        critique_prompt = f"""
你是一个专业的问题质量评估专家，专注于评估面向运维人员的技术问答数据质量。
请根据以下参考上下文和生成的问题，从8个维度对问题进行批评和评估。

## 评估维度

### 1. 元信息检查（Meta-info Avoidance）
检查问题是否包含不应该出现的文档结构引用：
- ❌ 步骤引用：如"在步骤6中"、"执行步骤4时"
- ❌ 章节引用：如"根据第三章"、"在3.2节中"
- ❌ 行号引用：如"第24行"、"代码第10行"
- ❌ 图表引用：如"根据流程图"、"表格中显示"
- ❌ 文档引用：如"根据提供的参考内容"、"原文中提到"、"根据告警参数"

### 2. 具体性平衡（Specificity Balance）
检查问题的具体程度是否适中：
- ❌ 过于细节：问到某个具体字段名、某个SQL语句中的参数、某个脚本的某一行
  - 反例："elp字段的含义是什么？"、"pg_replication进程的作用？"
- ❌ 过于宽泛：问题太泛，像摘要不像问题，无法明确回答
  - 反例："GaussDB集群无法使用怎么办？"（原因太多，无法针对性回答）
- ✅ 适中：有明确的问题范围，但不会细到某个字段/参数级别

### 3. 运维视角（Operator Perspective）
检查问题是否符合运维人员的知识背景、提问习惯以及真实的生产环境场景（如 Incident 排障、Change 变更、Configuration 配置等）：
- ❌ 脱离运维场景/缺乏环境约束：问题没有实际应用的业务背景，像是"为了提问而提问"（如"最佳实践是什么"却不带任何约束）
- ❌ 不符合运维人员知识水平：预设上帝视角（例如提问包含排障前不可能知道的底层原因），或问题假定已知答案
  - 反例："当网络分区导致 vote_timeout 时现象是什么？"（真实运维只能看到现象，不知根因）
- ❌ 常识性问题：问纯文档概念背诵、基础组件作用等，对排障决策无帮助
  - 反例："为什么集群需要有CN节点？"
- ❌ 与领域无关：问题与主系统运维无关，属于通用基础知识
- ✅ 场景贴合：紧扣真实的排障（Diagnostic）、变更操作（Procedural）、根因分析（RootCause）或配置说明（ConfigExample），基于可见现象或明确的运维目标发起

### 4. 表述清晰度（Clarity）
检查问题表述是否清晰无歧义：
- ❌ 有歧义：可能有多种理解，如数据盘还是日志盘
- ❌ 缺少上下文：问题没头没尾，需要补充信息才完整
  - 反例："哪里的24行？"、"什么步骤6？"
- ❌ 问题已包含答案：问题本身已经把答案说出来了
- ✅ 清晰完整：问题表述清晰，能明确理解意图

### 5. 实用价值（Practical Value）
检查问题对实际运维排障是否有帮助：
- ❌ 无实用价值：问题对解决实际问题没有帮助
  - 反例："界面上用什么颜色表示告警？"
- ❌ 不可操作：问题答案没有固定值，如"端口号是多少"（实际是配置决定的）
- ✅ 有价值：问题能帮助运维人员解决实际问题

### 6. 原文支撑（Source Alignment）
检查问题是否能在原文中找到对应内容：
- 无直接对应：原文没有专门说明该内容，需要从多处总结
- 完全无关：问题与原文内容完全无关
- 有支撑：问题在原文中有明确的对应内容

### 7. 回答边界与可验证目标（Answerable Boundary）
检查问题是否有明确的回答边界和可验证的落点：
- ❌ 开口式问法："请介绍 / 请总结 / 有哪些 / 如何评价 / 怎么看"等，没有具体落点
- ❌ 无边界的"最佳实践"：如"X 的最佳实践是什么"，未带维护窗口 / RPO-RTO / 版本 / 数据规模等任一约束
- ❌ FaultReproFix 类问题若未出现"测试 / 预发 / 演练 / 实验环境"等限定词，视为鼓励生产破坏性动作，不合格
- ✅ 有边界：问题能用一个明确答案回答，可被复核或被验证（例如：目标命令、阈值判断、决策选择、机制解释）

### 8. 答案泄漏检查（Answer Leakage）
检查问题本身是否已经把标准答案或核心结论写进题干：
- ❌ 题干直接包含参数值、命令名、阈值、结论：如"为什么把 innodb_buffer_pool_size 设置为物理内存的 70% 是合理的"（答案值已写进题干）
- ❌ 用因果陈述伪装提问：如"X 是因为 Y 才发生的，请解释 Y"
- ✅ 题干仅给出现象/约束，答案需要通过原文推断或查表才能给出

## 参考上下文
{context}

## 待评估的问题
{question}

## 输出格式要求
请严格按照以下格式输出你的评估：

[Critique Start]

[Meta Info Check Start]
评分：通过/不通过
说明：（如有元信息引用，请具体指出）
[Meta Info Check End]

[Specificity Check Start]
评分：适中/过于细节/过于宽泛
说明：（评估问题的具体程度）
[Specificity Check End]

[Operator Perspective Check Start]
评分：符合/不符合
说明：（评估是否符合运维人员提问习惯）
[Operator Perspective Check End]

[Clarity Check Start]
评分：清晰/不清晰
说明：（评估表述是否清晰无歧义）
[Clarity Check End]

[Value Check Start]
评分：有价值/无价值
说明：（评估对实际排障的帮助）
[Value Check End]

[Source Alignment Check Start]
评分：有支撑/无支撑
说明：（评估原文是否有对应内容）
[Source Alignment Check End]

[Answerable Boundary Check Start]
评分：通过/不通过
说明：（评估是否有明确回答边界和可验证落点；BestPractice 类是否含显式约束；FaultReproFix 类是否用测试/演练语境）
[Answerable Boundary Check End]

[Answer Leakage Check Start]
评分：通过/不通过
说明：（评估题干是否已把参数值/命令名/阈值/结论写进去）
[Answer Leakage Check End]

[Overall Assessment Start]
总体评估：需要改进/无需改进
问题数量：（指出有几个维度不通过）
[Overall Assessment End]

[Suggestion Start]
如需改进，请给出具体的改进建议，说明应该如何修改问题
[Suggestion End]

[Critique End]

请开始你的评估：
"""
        return critique_prompt


@PROMPT_REGISTRY.register()
class QuestionRefinePrompt(PromptABC):
    """
    # 用途：基于批评反馈改进生成的问题
    # 应用场景：在 QuestionRefiner 算子中使用（第二步）
    #
    # 改进原则：见 build_prompt 正文（10 条维度主干，其中 9/10 为最高优先级；
    # 另有第 11 条「保留题目性质」与第 12 条「保留有支撑的场景框架」）。
    #
    # 输入：
    # - context：参考上下文
    # - question：原始问题
    # - critique：批评反馈
    #
    # 输出格式：
    # [Improved Question Start]改进后的问题[Improved Question End]
    """
    def __init__(self):
        # 优化示例列表 - 按问题类型分类
        self.examples = [
            # 类型1：去除元信息引用（步骤、文档引用）
            {
                "original": "如果DN组件修复失败，在执行步骤6时需要收集哪些关键日志和信息用于技术支持分析",
                "refined": "如果DN组件修复失败，需要收集哪些关键日志和信息用于技术支持分析",
                "problem_type": "包含步骤引用"
            },
            {
                "original": "根据告警参数，哪些信息可用于定位产生告警的具体节点和实例？",
                "refined": "哪些信息可用于定位产生告警的具体节点和实例？",
                "problem_type": "包含文档引用"
            },
            # 类型2：问题过于细节 -> 提升到合适粒度
            {
                "original": "GaussDB实例中LOCALSSD类型的磁盘空间大小有哪些具体数值?",
                "refined": "GaussDB实例中LOCALSSD类型的磁盘中，根据不同分类用途，有哪些默认空间大小？",
                "problem_type": "问题过于细节"
            },
            # 类型3：问题转向可操作的场景
            {
                "original": "如果某节点磁盘IO带宽占用率达到96%，持续2分钟，是否会触发5023120告警？依据是什么",
                "refined": "某节点磁盘IO带宽占用率超过95%，触发5023120报警，请问该如何处理？",
                "problem_type": "问题已包含答案，转向可操作问题"
            },
            # 类型4：问题表述优化
            {
                "original": "为什么内核执行gs_replace命令修复节点会失败并导致集群不可用风险",
                "refined": "执行gs_replace命令修复节点失败的可能原因有哪些？",
                "problem_type": "表述优化，更像人类提问"
            },
            # 类型5：常识性问题 -> 转向实际场景
            {
                "original": "GaussDB集群中CN节点的作用是什么？",
                "refined": "CN节点故障会导致哪些问题？如何快速恢复？",
                "problem_type": "常识性问题转向实际场景"
            },
            # 类型6：问题过于宽泛 -> 聚焦具体场景
            {
                "original": "GaussDB集群无法使用怎么办？",
                "refined": "GaussDB集群连接超时，如何排查网络和实例状态？",
                "problem_type": "问题过于宽泛，需要聚焦"
            },
            # 类型7：最佳实践无约束 -> 补显式约束
            {
                "original": "GaussDB 主备集群有哪些最佳实践？",
                "refined": "在维护窗口只有 30 分钟、RPO 要求≈0 的主备场景下，如何规划同步提交策略？",
                "problem_type": "BestPractice 无约束，需补维护窗口/RPO/版本等显式约束"
            },
            # 类型8：故障复现未带测试语境 -> 明确限定环境
            {
                "original": "如何人为制造 DN 主备脑裂以观察切换行为？",
                "refined": "在测试环境中如何复现 DN 主备脑裂以验证切换行为与告警链路？",
                "problem_type": "FaultReproFix 必须带测试/演练语境"
            },
            # 类型9：题干泄漏答案 -> 去掉答案关键词
            {
                "original": "为什么把 max_connections 设置为 800 能避免连接风暴？",
                "refined": "当观察到连接风暴并伴随登录拒绝时，应如何评估并调整 max_connections？",
                "problem_type": "题干已泄漏参数值，改为现象+决策问法"
            },
            # 类型10：机制/根因类问题（答案以机制解释为主）
            {
                "original": "慢 SQL 是怎么产生的？",
                "refined": "在本版本中观察到大量慢 SQL 伴随 buffer_hit 偏低，从锁 / 执行计划 / 资源调度三个层面可能的根因有哪些？",
                "problem_type": "RootCause 类，需绑定现象和机制维度"
            },
            # ========== 反例：这些 "改写方向" 是禁止的（违反事实闭包 / 粒度一致） ==========
            # 反例 A（#48 风格：多步操作被窄化为单数值题）
            {
                "original": "脚本应包含哪些关键检查点和超时控制逻辑?",
                "refined": "（保持原粒度，仅可删除元信息引用）脚本应包含哪些关键检查点和超时控制逻辑？",
                "problem_type": "❌ 错误改写示例：把 5 步脚本问题窄化为 '内存不低于多少？等多久？' 会丢弃步骤 3/4/5；refine 不得引入 answer 中没有的具体数值，也不得抛弃 answer 已经覆盖的步骤。当 answer 是多步操作时，Q 必须覆盖所有步骤"
            },
            # 反例 B（#73 风格：路径枚举被下钻到字段级）
            {
                "original": "排查 vote_timeout 类故障时，应优先检查哪些日志路径？",
                "refined": "排查 vote_timeout 类故障时，应优先检查哪些日志路径及其功能描述？",
                "problem_type": "❌ 错误改写示例：把 '检查哪些日志路径' 下钻为 '检索哪些字段组合，列出字段名及典型值模式如 event=vote_timeout' 是禁止的——若 answer 只是 4 条路径枚举，refine 不得引入 answer 之外的字段名/值模式。当 answer 是路径枚举时，Q 的回答边界就是 '列出哪些路径'"
            },
            # 反例 C（#75 风格：refine 把 answer 中的路径替换成另外的路径）
            {
                "original": "应该重点检查 Other\\\\clock 和 Other\\\\devm_log 哪些日志？",
                "refined": "应该重点检查 Other\\\\clock 和 Other\\\\devm_log 哪些日志？",
                "problem_type": "❌ 错误改写示例：把 answer 提到的 Other\\\\clock 和 Other\\\\devm_log 替换为 Messages\\\\message_euler 和 Messages\\\\sys_logs_indisk 是禁止的；refine 只允许保留或删除 answer 已有路径，不得替换为 answer 未出现的路径"
            }
        ]

    def _format_examples(self) -> str:
        """格式化示例列表"""
        examples_text = ""
        for i, ex in enumerate(self.examples, 1):
            examples_text += f"""
示例{i}（{ex['problem_type']}）：
原问题：{ex['original']}
优化后：{ex['refined']}
"""
        return examples_text

    def build_prompt(self, context: str, question: str, critique: str, answer: str = "") -> str:
        examples_text = self._format_examples()
        answer_section = (
            f"\n## 参考答案（事实闭包校验的第一参照；精炼后的问题约束必须能从此答案直接推出）\n{answer}\n"
            if answer and answer.strip()
            else ""
        )

        refine_prompt = f"""
你是一个专业的问题优化专家，专注于优化面向运维人员的技术问答数据。
请根据批评反馈和以下原则改进问题质量。

## 改进原则（10 个维度，9 / 10 为最高优先级，与其他原则冲突时优先）

### 1. 去除元信息引用
- 删除所有对"步骤X"、"第X章"、"第X行"的引用
- 删除"根据流程图"、"根据表格"等图表引用
- 删除"根据提供的参考内容"、"原文中提到"等文档引用

### 2. 平衡具体性
- 如果问题过于细节（问某个字段、某个参数），提升到合适的粒度
- 如果问题过于宽泛，聚焦到具体的故障场景或操作步骤

### 3. 符合运维视角与真实场景约束
- 问题必须贴合真实的运维/排障/变更/调优场景，像运维人员（SRE/DBA）在遇到实际生产问题时的提问
- 避免常识性或纯文档概念的背诵问题，强制转向具体的现象排查、操作指南或根因分析
- 确保提问者的视角是"从事前现象出发求解决"，而非"带着事后结论反推现象"
- 如果是"最佳实践"或"故障复现"，必须补齐明确的环境约束（如维护窗口、版本、资源限制、演练环境等）

### 4. 表述清晰
- 消除歧义，确保问题只有一种理解方式
- 补充必要的上下文，使问题完整
- 如果问题已包含答案，转向询问处理方法

### 5. 保持实用价值
- 确保问题对实际运维排障有帮助
- 避免问没有固定答案的配置值

### 6. 确保原文支撑
- 问题应该能在原文中找到对应的答案
- 如果原文无法支撑，考虑调整问题范围

### 7. 明确回答边界与可验证目标
- 删除"请介绍 / 请总结 / 有哪些 / 如何评价"这种开口式问法，替换为现象+决策型问法
- BestPractice 类问题必须带至少一条显式约束从句（维护窗口 / RPO-RTO / 版本 / 数据规模 / 预算 等任一）
- FaultReproFix 类问题必须出现"测试 / 预发 / 演练 / 实验环境"之类的限定词，禁止鼓励生产环境破坏性动作

### 8. 去除答案泄漏
- 题干中不得出现标准答案的参数值、命令名、阈值、结论
- 若原题已把答案值写进来（例如"把 X 设成 70% 是否合理"），改写为"当观察到现象 Y 时如何评估并调整 X"

### 9. 事实闭包（与参考答案对齐，最高优先级，冲突时本条压倒其他）
- 精炼后的问题所有限定条件（实体 / 参数 / 文件名 / 字段名 / 数值 / 场景现象）**必须全部出现在当前 answer 文本或提供的 context 中**；不得引入 answer 未涉及的新事实
- 禁止在问题里新增**占位式具体值示例**（例如 `event="vote_timeout" AND node_id="0x1234"`）当作"更具体"的装饰；任何带引号 / 反引号的具体值都必须来自 answer/context 原文
- 禁止替换 answer 中提到的路径 / 文件 / 命令为其他路径 / 文件 / 命令；refine 只允许**保留或删除** answer 里已有的，不允许**替换**
- 自检：把精炼后的 Q 给只看 answer 的人，他能不能从 answer 文本直接得出结论？不能则 refine 无效，回退到原问题

### 10. 粒度一致性
- 若 answer 是单句事实陈述（≤ 2 句），Q 必须是"事实确认型"，禁止抬高为"字段级 / 步骤级 / 验证方案"
- 若 answer 是路径对照表 / 配置项枚举，Q 的回答边界应为"列出哪些路径 / 哪些项"，不得下钻到"字段名 / 值模式 / 关键字"
- 若 answer 是多步操作 / 脚本框架，Q 必须覆盖所有步骤，不得窄化为只问其中一两个数值

### 11. 保留题目性质（不要越界改题型）
- 若原题是 RootCause / BestPractice / ComplianceSecurity 等以"机制解释 / 决策依据 / 合规说明"为交付物的问题，不要强行改写成"给命令"的操作题
- 改写只能收紧约束、澄清表述，不得改变题目本质的交付物类型

### 12. 保留有支撑的场景框架（防止退化成文档背诵题）
- 原问题中的真实运维场景（例如升级后、开箱验货、硬件环回测试、OTDR 测试、链路闪断、dmesg 出现 link down、静电敏感区操作等）不是冗余背景；只要这些场景元素能在 answer 或 context 中找到支撑，改写后必须尽量保留
- 可以删除原问题中**答案无法覆盖**的额外任务（例如“如何验证是否受损”“如何设计闭环流程”），但不要把整个场景删成纯事实背诵题
- 若原问题是“在场景 S 下，应该做/避免/确认什么”，改写后仍应是“在场景 S 下，应该做/避免/确认什么”，而不是泛化为“X 的要求是什么”
- 若必须删除某个场景元素，必须是因为它不在 answer/context 中、泄漏答案、或会导致答案无法覆盖；否则视为改写失败，应保持原问题

## 优化示例
{examples_text}
{answer_section}
## 参考上下文
{context}

## 原始问题
{question}

## 批评反馈
{critique}

## 输出格式要求
请严格按照以下两个区块依次输出（顺序不可调换，缺一不可）：

[Analysis Start]
简要分析原问题的主要问题（1-2句话）
[Analysis End]

[Improved Question Start]
你改进后的问题
[Improved Question End]

[Fact Closure Check Start]
```json
{{
  "all_constraints_in_answer_or_context": true,
  "new_facts_introduced": [],
  "replaced_facts": [],
  "granularity_match": "事实陈述 / 路径枚举 / 多步操作 / 机制解释 / 根因分析",
  "scenario_preservation": {{
    "kept": ["保留的原问题场景元素"],
    "removed": ["删除的场景元素及原因"]
  }}
}}
```
[Fact Closure Check End]

字段说明：
- `replaced_facts`：精炼后问题用 X 替换了 answer 中的 Y 的"X→Y"对（仅当替换发生时填，否则空数组）
- `granularity_match`：判断 answer 的粒度类别，从 ["事实陈述", "路径枚举", "多步操作", "机制解释", "根因分析"] 中选一项
- `scenario_preservation`：说明保留了原问题中哪些有支撑的场景元素；若删除场景元素，必须说明是无支撑 / 泄漏答案 / 答案无法覆盖

注意：
1. **优先保持事实闭合：尽量避免新增事实与替换事实；若为提升可答性必须调整表达，仅可做可在 answer 或 context 中定位的等价改写，并在检查项中如实记录**
2. **优先保留有支撑的场景元素：若场景信息冗长、弱相关或易泄漏答案，可适度压缩；仅在无法被 answer 或 context 支撑时删除，并在 scenario_preservation.removed 中说明原因**

请开始改进：
"""
        return refine_prompt


@PROMPT_REGISTRY.register()
class AnswerCritiquePrompt(PromptABC):
    """
    # 用途：对生成的答案进行批评，指出答案的优缺点
    # 应用场景：在 AnswerRefiner 算子中使用（第一步）
    #
    # 批评维度（8个）：
    # 1. 完整性
    # 2. 准确性
    # 3. 针对性
    # 4. 结构清晰度
    # 5. 原文支撑
    # 6. 元信息规避
    # 7. 可操作性门槛（类别适用时）
    # 8. 结构压缩约束
    #
    # 输入：
    # - context：参考上下文
    # - question：用户问题
    # - answer：生成的答案
    #
    # 输出格式：
    # [Critique Start]
    # [各维度检查结果]
    # [Overall Assessment]
    # [Suggestion]
    # [Critique End]
    """
    def __init__(self):
        pass

    def build_prompt(self, context: str, question: str, answer: str) -> str:
        critique_prompt = f"""
你是一个专业的答案质量评估专家，专注于评估面向运维人员的技术问答数据质量。
请根据以下参考上下文、问题和生成的答案，从8个维度对答案进行批评和评估。

## 评估维度

### 1. 完整性（Completeness）
检查答案是否包含了原文中的所有关键步骤/要点：
- ❌ 不完整：原文有10个处理步骤，但答案只提到1-2个
- ❌ 遗漏关键信息：缺少重要的前置条件、注意事项或后续步骤
- ✅ 完整：包含了原文中与问题相关的所有关键信息

### 2. 准确性（Accuracy）
检查答案内容是否正确：
- ❌ 事实错误：答案中的信息与原文不符
  - 反例：原文说默认用户是bss_admin，答案却说是root
- ❌ 编造固定值：给出原文没有的固定端口号、固定挂载点等
  - 反例："端口号是8080"（实际是配置决定的）
- ❌ **Shell/命令扩写编造**：参考答案仅含一条简单命令或简短说明，答案却写出多行 `for`/`while`、`awk`、`grep -E` 等组合脚本且原文无此类结构——判不准确
- ❌ 有乱码或格式错误
- ✅ 准确：答案内容与原文一致，无事实性错误

### 3. 针对性（Relevance）
检查答案是否直接回答了问题：
- ❌ 答非所问：问原因但答处理方法，或反之
  - 反例：问"为什么会失败"，答"按以下步骤处理"
- ❌ **整段抄原文（answer-as-quote）**：把参考上下文中与问题**仅弱相关**的大段原文逐条列出，却**不先给出针对问题的结论**（例如问「仅有乙醇和棉签能否清洁光纤接头」，却罗列「必须使用专用溶剂、无纺纸、压缩气…」而不先答能/不能及条件）——判「针对性弱」，REWRITE 时须**先直接回应问题**，再按需极简引用支撑句，禁止无差别 dump 段落
- ❌ **问防/问禁却答违规清单或答反**：问「应做何种防护」却答「当前流程中可能存在的违规点」；问「必须对哪类连接执行何种操作」却用「该操作违反了…」开头且**无指代**；问「**应避免**哪项操作」却用「✅ 优先操作：…」或只写「应优先…」——判不通过，须改为与问题**极性一致**的直接答案（避免什么 / 禁止什么 / 必须先做什么）
- ❌ **评判/排序语言泄漏**：答案使用「最可能」「应优先（而无明确优先级语境）」「当前维护流程中可能存在以下违反…」等**出题或评估口吻**，而非运维答复口吻——视为不通过，须改写为直白陈述
- ❌ **脱离运维实操**：针对 Incident排障、Change变更等明确场景问题，答案却停留在系统理论解释，未提炼出实质的排查路径、操作指标或判断标准
- ❌ 过于宽泛：没有针对性，泛泛而谈
- ❌ 偏离焦点：答案涉及很多不相关的内容
- ❌ 场景推断溢出（Scenario Inference Overflow）：问题是"X 和 Y 分别记录什么信息"或"哪些路径用于排查 Z"这类**纯事实/枚举型查询**，但答案末尾追加"升级后二者异常，表明 XX 可能中断…这直接影响 YY"或"该现象意味着……可能引发……"这类**自行扩写的因果推断 / 影响分析 / 后续风险判断段**——视为不通过，REWRITE 时必须删除该推断段，仅保留问题明确询问的事实
- ❌ **答案不可脱离上下文独立成立**：若问题表述极泛（例如「重点确认哪两类状态」），而答案只是同义空话、未给出可核验的条目，视为针对性弱（除非参考答案本身已锁住两类具体状态名称）
- ❌ **演练剧本灌水**：问题仅问访问地址或下载路径，答案却写成多角色长篇演练对白——判不通过，须改为事实句
- ✅ 针对性强：直接回答问题，与问题焦点一致，不夹带未询问的推断

### 4. 结构清晰度（Structure）
检查答案的格式和可读性：
- ❌ 格式混乱：全部堆在一段，没有分点分段
- ❌ 逻辑不清：答案缺乏层次，难以理解
- ❌ 可读性差：没有使用列表、编号等结构化格式
- ✅ 结构清晰：分点/分段，有条理，易于阅读

### 5. 原文支撑（Source Fidelity）
检查答案是否基于原文：
- ❌ 无原文支撑：答案内容在原文中找不到依据
- ❌ 模型编造：答案是模型自己的能力输出，非原文提供
  - 反例：原文没有"临时性和永久性"的说法，答案却使用了
- ❌ 超出原文范围：回答了原文没有涉及的内容
- ❌ **复杂 shell 仅出现在答案中**：参考答案无 `awk`/`grep -E`/多行 `for …; do`，答案却出现——高度疑似编造，判无支撑
- ✅ 有支撑：答案的每个要点都能在原文中找到依据

### 6. 元信息检查（Meta-info Avoidance）
{META_FORBIDDEN_ZH_CRITIQUE_BRIEF}
- ❌ **装饰性栏目与 emoji 标签**：行首 `✅`/`⚠️` 加「优先操作 / 验证方法 / 局限说明」等伪 UI 标签未清理——判不通过，REWRITE 时删除装饰符，改为纯文本分点

### 7. 可操作性硬门槛（Executability Gate，针对诊断排查 / 报错处理 / 高可用应急 / 性能调优类问题）
对上述类别的问题，以下两项必须同时满足，任意失败即判"不通过"并要求 REWRITE：
- `has_executable_step`：答案中至少包含一条可直接执行的命令、SQL、系统视图查询、配置项修改或检查动作
- `step_is_complete`：上述命令/步骤必须具备直接落地的完整度，即参数/对象/前置条件齐全，值班人员可以复制即用或照做，而不是伪代码或占位描述
但本门槛**不得诱导编造**：
- 若参考上下文没有提供任何命令、路径、SQL、配置项、系统视图或可执行检查动作，不得在建议中自行发明 `ls` / `grep` / SQL / 路径示例
- 这种情况下应在 Source Fidelity / Relevance 中说明“当前材料不足以支撑该操作型问题”，并建议改为参考上下文能够支撑的最小答案；不得靠 AnswerRefiner 补写原文没有的操作细节
- 只有当参考上下文本身提供了可执行动作或足够明确的检查对象时，才要求 REWRITE 补齐步骤
以下两类问题本条不适用，标记为 N/A，不强制要求命令或 step_is_complete：
- (a) 纯知识记忆型：仅问定义 / 参数名 / 默认值 / 概念边界
- (b) 机制-决策-合规型：RootCause（根因机制） / BestPractice（选型依据与权衡） / ComplianceSecurity（权限、加密、审计、合规说明） / 差异对比结论等，答案以解释或决策依据为主，命令不是主要交付物

### 8. 结构压缩约束（Structure Compression）
检查答案是否存在结构膨胀，以下任一命中即判"不通过"：
- ❌ 出现三级及以上嵌套（例如同时使用小标题 + 编号 + 子项缩进 + 二级无序列表）
- ❌ 同一条答案混用多种主组织形式（短段落、编号步骤、无序列表三选一，不得混用）
- ❌ 出现与问题无关的铺垫段、总结段或礼貌语

## 参考上下文
{context}

## 用户问题
{question}

## 待评估的答案
{answer}

## 输出格式要求
请严格按照以下格式输出你的评估：

[Critique Start]

[Completeness Check Start]
评分：完整/不完整
说明：（评估答案是否包含所有关键信息）
[Completeness Check End]

[Accuracy Check Start]
评分：准确/不准确
说明：（评估答案是否有事实性错误或编造内容）
[Accuracy Check End]

[Relevance Check Start]
评分：针对性强/针对性弱
说明：（评估答案是否直接回答问题）
[Relevance Check End]

[Structure Check Start]
评分：清晰/不清晰
说明：（评估答案格式和可读性）
[Structure Check End]

[Source Fidelity Check Start]
评分：有支撑/无支撑
说明：（评估答案是否基于原文）
[Source Fidelity Check End]

[Meta Info Check Start]
评分：通过/不通过
说明：（如有元信息引用，请具体指出）
[Meta Info Check End]

[Executability Check Start]
适用性：适用/不适用（若问题为纯知识记忆型，请填"不适用"）
has_executable_step：通过/不通过
step_is_complete：通过/不通过
说明：（指出是否给出可直接复制执行的命令/SQL/检查动作，步骤是否完整闭合）
[Executability Check End]

[Structure Check2 Start]
评分：通过/不通过
说明：（检查是否出现三级嵌套、多种主组织形式混用、与问题无关的铺垫段/总结段/礼貌语；答案长度不设固定字数上限，以原文支撑的关键信息是否完整呈现为准）
[Structure Check2 End]

[Overall Assessment Start]
总体评估：需要改进/无需改进
问题数量：（指出有几个维度不通过）
强制重写（Force Rewrite）：是/否（当问题属于诊断排查 / 报错处理 / 高可用应急 / 性能调优类，且 Executability 或 Structure Check2 任一为"不通过"时，必须填"是"）
[Overall Assessment End]

[Suggestion Start]
如需改进，请给出具体的改进建议，说明应该如何修改答案。
要求：建议本身也必须有原文支撑，不得给出参考上下文没有出现的命令、路径、字段、阈值、目录结构、验证方法示例；如果某条要求受原文限制无法完全满足，请明确指出可安全执行的删改动作（例如删除无支撑断言、删除文档坐标、改成“当前材料仅能确认……”），不要编造示例。
[Suggestion End]

[Critique End]

请开始你的评估：
"""
        return critique_prompt


@PROMPT_REGISTRY.register()
class AnswerRefinePrompt(PromptABC):
    """
    # 用途：基于批评反馈改进生成的答案
    # 应用场景：在 AnswerRefiner 算子中使用（第二步）
    #
    # 改进原则要点：参考答案封闭为第一优先级；其下为多维度编排（完整性、准确性、
    # 针对性、结构、元信息去除、可操作性软约束、结构压缩等，详见 build_prompt）。
    #
    # 输入：
    # - context：参考上下文
    # - question：用户问题
    # - answer：原始答案
    # - critique：批评反馈
    #
    # 输出格式：
    # [Improved Answer Start]改进后的答案[Improved Answer End]
    """
    def __init__(self):
        pass

    def build_prompt(self, context: str, question: str, answer: str, critique: str) -> str:
        refine_prompt = f"""
你是一个专业的答案优化专家，专注于优化面向运维人员的技术问答数据。
请根据批评反馈和以下原则改进答案质量。

## 第一优先级：原文封闭约束（最高优先级，与任何其他原则冲突时本条优先）

「参考上下文」是**唯一被许可的事实来源**；「参考答案」是已有作答，作为改写的起点而非信息天花板。你的改写必须满足：
1. refined_answer 的**信息集**（包括命名实体、参数值、命令、步骤、阈值、验证方法、机制解释、因果链条、范围限定）**必须是参考上下文信息集的子集**
2. 参考答案是改写起点：若批评指出参考答案遗漏了参考上下文中有明确支撑的信息，允许从参考上下文补充；若参考答案已完整准确，应保持与参考答案高度一致，不做不必要的改动
3. 不得基于常识、经验、行业最佳实践**自行补充**命令、数值、步骤、机制解释、验证方案、根因分析
4. 若参考答案只是一句事实陈述（例如"设备上有 X 标识，提示 Y"），且参考上下文也无更多支撑，必须保持为**事实陈述形式**，不得扩写成机制分析 / 验证方案 / 根因分析 / 多步操作指南
5. 允许的改写动作：**重组结构、分点排版、合并同义句、删除冗余、纠正错别字、替换元信息引用、轻度语言润色；当批评指出明确问题时，允许进行针对性实质重写**
6. 禁止无依据的"整段换皮重写"：不得在批评无明确指向的情况下整段替换参考答案；尤其不得把短事实陈述扩写成长篇解释段落
7. **删减式修复例外**：当批评反馈明确指出答案中存在「无原文支撑 / 元信息引用 / 答非所问 / 问题要求超出材料」时，允许删除这些有问题的句子或收缩成更保守的表述；这种情况下不要求保留多数原文字面，但仍然不得新增参考上下文之外的信息

## 最小代价原则（Least-Change Principle）

优先选择"**对参考答案做最小修改**"——只在批评反馈指出明确问题时才改动。如果参考答案本身已经干净、对齐问题、可读，**refined_answer 应该与参考答案几乎一致**（仅排版差异），不必为了"优化"而重写。

## 第二优先级：批评反馈对齐（次优先级，以不违反第一优先级为前提）

批评反馈中的**各维度检查说明**不是装饰，也不是只看 Suggestion 总结。你必须逐项读取 Completeness / Accuracy / Relevance / Source Fidelity / Meta Info / Executability / Structure 等失败维度，把其中可安全执行的要求转化为改写动作：
- 对“无支撑 / 模型编造 / 原文未提供”的内容：删除或改成**无前缀套话**的最小事实陈述；`refined_answer` **正文禁止**出现「当前材料」「参考上下文」「参考内容」等材料边界用语（与 §6 `META_FORBIDDEN` 一致，用直接事实句替代）
- 对“元信息引用”：删除文档页码、章节、步骤等出处坐标，不能用另一种元信息短语替换
- 对“答非所问”：先回答问题能被参考答案支撑的部分；不能回答的部分不要硬补
- 对“缺少命令 / 验证步骤”：只有参考答案或参考上下文已提供命令、路径、检查对象时才补；否则不要编造命令

当批评反馈的某条建议会导致你**引入参考上下文之外的内容**（例如要求"补充验证命令"、"补齐根因分析"、"增加前置条件"，但参考上下文中均无对应内容）时：
- **必须拒绝该条建议**，在 Analysis 段落里注明 "该建议超出参考上下文范围，忽略"
- 不得用编造的内容满足这条建议；但仍要执行该建议中**不引入新事实**的部分，例如删除无支撑断言、删除元信息、把答案收缩为不带材料套话的保守表述（例：直接写「仅能确认 X；所述范围内未给出 Y」，勿写「当前材料…」「参考上下文…」）

## 改进原则（以下所有原则都受第一优先级约束）

### 1. 确保完整性（以参考上下文为准）
- 可在参考上下文有明确支撑的前提下，补齐参考答案遗漏的要点
- 不得以"完整性"之名引入参考上下文没有的内容

### 2. 保证准确性
- 核对答案中的信息是否与参考上下文一致
- 删除任何编造的固定值（如端口号、路径等）
- 修正错别字、乱码、格式错误

### 3. 提高针对性
- 确保答案紧扣参考上下文所支持的信息去回应问题
- 若问题问的粒度超出参考上下文的支撑范围（例如问"根因"但原文只给事实），**保持原文支撑的粒度作答**，不得为了对齐问题而编造
- **强化运维实操价值**：若处理排障（Incident）、变更（Change）等明确场景，优先提炼具有执行价值的判断指标或排查路径，剔除宽泛的理论铺垫

### 4. 优化结构
- 将长段落拆分为分点或分步骤
- 使用编号列表使步骤更清晰
- 保持逻辑层次分明，但不得超过两级嵌套

### 5. 忠于原文（等价于第一优先级）
- 只使用参考上下文中有明确支撑的信息
- 参考答案是改写起点，批评指出的遗漏可从参考上下文补充；不得超出参考上下文范围
- 如果参考上下文信息不足以回答问题，**宁可回答不完整，也不得编造**

### 6. 去除元信息引用（强制禁词清单，命中任意一条即整句删除或重写）
{META_FORBIDDEN_ZH_REFINE_BLOCK}

### 7. 可操作性（软要求，受第一优先级约束，不再是硬门槛）
- 若参考上下文包含命令 / SQL / 系统视图查询 / 参数值 / 检查动作，精炼时保持这些信息完整并补全前置条件 / 验证方式（前提：前置条件 / 验证方式也来自参考上下文）
- 若参考上下文**没有**任何可执行内容，**不得补命令**，即使问题属于诊断 / 报错 / 应急 / 调优类
- 以下类别的问题即使问得像"怎么做"，也可以只保持事实陈述或机制解释，不需要强行提供命令：
  (a) 纯知识记忆型：仅问定义 / 参数名 / 默认值 / 概念边界
  (b) 机制-决策-合规型：RootCause（根因机制） / BestPractice（选型依据与权衡） / ComplianceSecurity（权限、加密、审计、合规说明） / 差异对比结论等

### 8. 结构压缩约束
- 禁止出现三级及以上嵌套（例如同时有小标题 + 编号 + 子项缩进）
- 单条答案只允许一种主组织形式：短段落、编号步骤、无序列表三选一，不得混用
- 删除与问题无关的铺垫段、总结段、礼貌语和冗余背景介绍
- ❌ **禁止保留问题未要求的场景推断段**：若 refined_answer 包含问题未询问的因果推断 / 影响分析 / 后续风险判断段落（例如问"X 和 Y 分别记录什么"，但答案末尾追加"升级后二者异常表明 XX 可能中断…这直接影响 YY"），必须删除该推断段，仅保留问题明确询问的信息

### 9. 反向表述规范（直接陈述，禁止"应检查/应关注"型回避句式）
- 当问题属于**枚举型**（"需要填哪些参数 / 包含哪些字段 / 记录哪些信息 / 输出哪些列"），refined_answer 必须采用**直接陈述语气**列出条目
- ❌ 禁止反向表述：`应检查并重新确认 XX 是否正确` / `应关注 XX` / `需要注意 XX` / `建议核对 XX`
- ✅ 要求直接陈述：`需要填写的参数包括：XX、YY、ZZ` / `记录的信息为：A、B、C`

### 10. 输出禁内部标签词（防止内部术语泄漏）
**严禁**在 [Improved Answer Start]/[End] 区块内出现以下内部标签或工程化术语：
`事实锚` / `事实锚答案` / `事实基准` / `Fact Baseline` / `Target Answer` / `精炼答案` / `事实集封闭` / `Fact-Set Closure` / `参考答案` / `可用信息集` / `批评反馈` / `Critique` / `OUTOFSCOPE`。
若答案确实需要表达"原文未提及 X"，应改写为直接陈述"根据当前信息无法确认 X"，且这种 case 通常应由上游 DROP，不应走到本步。

### 11. 直接应答、抗「整段抄 input」与问答题极性对齐（RAG 训练友好）
- **禁止 answer-as-quote**：不得以「把参考正文相关段落整段贴出」代替作答；必须先**用一两句直接回应问题**（结论/步骤/判断），再仅在必要时引用原文要点，且不得抄与问题无关的禁令或背景（例：问清洁材料，未问眼睛防护则不得大段抄激光安全）
- **问什么答什么**：问「分别应执行什么防护」→ 答具体防护动作，不得改写成「可能存在哪些违规」；问「必须对哪类连接执行何种操作」→ 首句须点明**连接类型 + 操作**，禁止无指代地以「该操作违反了…」开头
- **极性一致**：问「应避免/禁止/切勿」时，不得只答「✅ 优先操作…」或「应优先…」而不提禁止项；若原文只有正面步骤，应明确写出「因此不应做…」或「须避免…」与问题对齐
- **禁评估腔**：禁止使用「最可能」「应优先检查以下…（而无明确排序依据）」「当前维护流程中可能存在以下违反…」等出题/评分用语；改为值班口径的直接陈述
- **禁剧本灌水**：问地址/路径/下载入口时，只输出事实信息，禁止扩写为多角色演练对白
- **禁装饰符与伪栏目**：输出中不得出现行首 `✅`/`⚠️` 以及「优先操作：」「验证方法：」「局限说明：」等训练污染标签（代码层会 strip，但模型仍不应产生）
- **禁扩写 shell**：参考上下文仅含单行或简短命令时，不得改写成含 `for`/`while`、`awk`、`grep -E` 的多行脚本；保持与原文同 grain 的命令粒度

## 参考上下文（事实来源；允许从中补充参考答案遗漏的、有原文支撑的内容）
{context}

## 用户问题
{question}

## 参考答案（改写起点；如批评指出遗漏，可从参考上下文补充；不得超出参考上下文范围）
{answer}

## 批评反馈（次优先级；要求引入参考上下文之外内容的建议一律忽略）
{critique}

## 输出格式要求
请严格按照以下格式输出：

[Analysis Start]
简要分析原答案的主要问题（1-2句话）。
若批评反馈中有建议会引入参考上下文之外的内容，请在这里点名拒绝并说明原因。
[Analysis End]

[Improved Answer Start]
你改进后的答案
[Improved Answer End]

硬性约束（违反任意一条视为本次改写不合格）：
1. 改进后的答案必须与原答案使用相同的语言
2. refined_answer 的信息集必须 ⊆ 参考上下文的信息集；可补充参考答案遗漏但有原文支撑的内容，不得引入参考上下文之外的常识或编造内容
3. 只允许一种主组织形式（短段落 / 编号步骤 / 无序列表），且不得出现三级及以上嵌套
4. 参考答案是简短事实陈述、且参考上下文也无更多支撑时，refined_answer 必须保持事实陈述形式，不得扩写成机制分析 / 验证方案 / 根因分析
5. 未在参考上下文中出现的命令、SQL、阈值、参数值、验证方法一律不得写入
6. 不得保留问题未要求的场景推断 / 影响分析 / 后续风险判断段落
7. 不得在输出中出现内部标签词（事实锚 / Target Answer / 参考答案 / 可用信息集 / 批评反馈 / Critique / OUTOFSCOPE 等）
8. 不得以"参考 / 参考文档 / 根据参考内容 / 原文指出 / 上述信息"等元信息短语开头或穿插，也不得保留任何"参考"、"文档"、"资料"等出处字眼，必须直接陈述事实
9. 不得使用行首 emoji 装饰符（✅⚠️ 等）或「优先操作：/ 验证方法：」等伪栏目式标签行
10. 问题问「避免/禁止」时，答案不得仅用「优先/应首先」作答而不体现禁止对象；问题问「能否」时须先给出可否判断
11. 不得引入参考上下文中未出现的多行 shell 结构（含 `for`/`while`、`awk`、`grep -E` 等）若参考上下文仅含简单命令或与这些 token 无关
12. 当批评反馈已指出答案中某些内容无支撑或属于元信息，必须删除/收缩这些内容；不要为了满足字符面重合而保留已被批评为错误的句子

请开始改进：
"""
        return refine_prompt


@PROMPT_REGISTRY.register()
class DistillQuestionGeneratorPrompt(PromptABC):
    """
    # 用途：基于标签/主题和参考上下文蒸馏生成高质量领域问题
    # 应用场景：在 DistillQuestionGenerator 算子中使用
    #
    # 核心功能：
    # 1. 根据给定的标签/主题生成领域相关问题
    # 2. 可基于参考上下文生成更具体的问题
    # 3. 按「运维过程 × 题型」双轴覆盖（10 类 ITIL/DevOps 场景 × 11 类题型）
    # 4. 强制占比契约，抑制纯定义背诵题与无边界开口题
    # 5. 支持客观题型：FillBlank（完形/填空）和 MultipleChoice（单选题）
    # 6. 支持 allowed_types 约束，配合文本信号分析两步出题
    #
    # 参数：
    # - count：生成问题的数量，默认 10
    # - existing_questions：已有问题列表，用于避免重复
    #
    # 特点：
    # - 问题必须与标签主题紧密相关
    # - 如有上下文，问题需基于上下文内容
    # - 场景 × 题型双轴覆盖，核心排障/变更类占比过半
    # - 事实型与最佳实践型必须带约束/现象，禁止退化为文档背诵
    # - 填空题/单选题含完整答案字段，适合直接用于训练
    # - 避免重复或高度相似的问题
    """

    # 优质种子问题示例（从 xlsx A 评分问题精选 + 新构造客观题，用于 few-shot 引导）
    # 覆盖：轴A（Incident/Problem/Change/Configuration/Security/Continuity/Capacity/Availability）
    #       轴B（Diagnostic/RootCause/Procedural/ComplianceSecurity/ConfigExample/Factoid/FillBlank/MultipleChoice）
    _SEED_EXAMPLES = [
        # ── 主观题型示例（来自 xlsx A 评分） ────────────────────────────────────────
        {
            "scenario": "Incident",
            "question_type": "Diagnostic",
            "question": (
                "GAUSS-20775 报错提示 \"the function or procedure with exception can't be pushed down "
                "for execution\"，当 SQL 中调用含 EXCEPTION 块的自定义函数或存储过程时触发。"
                "运维人员如何快速定位引发该错误的具体函数或存储过程？定位后应如何修改以满足下推执行要求？"
            ),
        },
        {
            "scenario": "Problem",
            "question_type": "Diagnostic",
            "question": (
                "GaussDB 集群中 pg_stat_activity 显示大量状态为 'idle in transaction' 的连接长期不释放，"
                "导致活跃连接数持续接近 max_connections 上限，应如何快速定位根源并清理？"
            ),
        },
        {
            "scenario": "Problem",
            "question_type": "RootCause",
            "question": (
                "当 VACUUM 执行后 pg_class.reltuples 统计值显著偏离实际数据量（如通过 COUNT(*) 验证），"
                "且 pg_stat_progress_vacuum.vm_scan_skipped > 0 时，该偏差最可能的根本原因是什么？"
                "请结合 vm_scan_skipped 字段的语义说明其与统计偏差的关联机制。"
            ),
        },
        {
            "scenario": "Change",
            "question_type": "RootCause",
            "question": (
                "walwriter_cpu_bind、walwriteraux_bind_cpu 和 wal_sender_bind_cpu_attr 这三个参数都涉及 CPU 绑定，"
                "它们各自绑定的对象、适用场景和配置注意事项有何关键区别？"
            ),
        },
        {
            "scenario": "Incident",
            "question_type": "Procedural",
            "question": (
                "GaussDB 中，若应用连接会话的 AUTOCOMMIT=ON，执行多条 DML 后未显式 COMMIT，"
                "但业务出现数据不一致或部分语句被意外提交，应如何快速确认当前会话的 AUTOCOMMIT 状态"
                "并安全调整为手动提交模式？"
            ),
        },
        {
            "scenario": "Continuity",
            "question_type": "Procedural",
            "question": (
                "当同城双中心容灾环境中 RPO 持续大于 0 时，应按什么顺序检查哪些关键状态？"
                "请列出必须满足的 3 个 RPO=0 前置条件（含 Dorado 共享 Xlog 盘状态、"
                "主实例归档状态、备实例恢复状态）及对应验证方法。"
            ),
        },
        {
            "scenario": "Security",
            "question_type": "ComplianceSecurity",
            "question": (
                "当仅授予用户对视图的 SELECT 权限时，该用户能否通过查询 information_schema.columns "
                "或执行 \\d+ view_name 获取其底层基表的字段名、数据类型等元数据？"
                "WITH (security_barrier) 是否能阻止此类元数据泄露？"
            ),
        },
        {
            "scenario": "Capacity",
            "question_type": "ConfigExample",
            "question": (
                "GaussDB 集群出现 XLOG 回收延迟告警，同时 pg_stat_bgwriter 显示 buffers_clean 持续偏低、"
                "dirty_page_percent_max 长期接近阈值，应如何协同调整 bgwriter_delay、"
                "candidate_buf_percent_target 和 pagewriter_sleep 参数以缓解脏页堆积？"
                "调整前后应检查哪些指标验证效果？"
            ),
        },
        {
            "scenario": "Incident",
            "question_type": "Factoid",
            "question": (
                "llvm_max_memory 参数与系统内存视图 GS_TOTAL_MEMORY_DETAIL 中的 llvm_used_memory 字段"
                "在运维层面存在哪些关键差异？为何调小 llvm_max_memory 无法立即降低 llvm_used_memory 的观测值？"
            ),
        },
        # ── 填空题示例（FillBlank）───────────────────────────────────────────────
        {
            "scenario": "Configuration",
            "question_type": "FillBlank",
            "question": (
                "walwriteraux_bind_cpu 参数类型为___，属于___类参数（修改后须___生效）；"
                "其实际有效上限为___，超过该值将导致___。"
            ),
            "blank_answer": "整型；POSTMASTER；重启数据库；CPU 核数减 1；数据库无法启动",
        },
        {
            "scenario": "Change",
            "question_type": "FillBlank",
            "question": (
                "为索引启用透明数据加密（TDE）时，WITH 子句中只允许设置___参数；"
                "若同时指定 encrypt_algo 或 key_type 等其他参数，将触发错误码___，"
                "系统给出的 CAUSE 为___。"
            ),
            "blank_answer": "enable_tde=on；GAUSS-81700；only 'enable_tde' can be set for index",
        },
        # ── 单选题示例（MultipleChoice）─────────────────────────────────────────
        {
            "scenario": "Incident",
            "question_type": "MultipleChoice",
            "question": "执行 CREATE INDEX 时触发 GAUSS-81700 报错，以下处理方式正确的是？",
            "options": [
                "A. 在 WITH 子句中同时保留 enable_tde=on 和 encrypt_algo 参数",
                "B. 只设置 enable_tde=on，删除 encrypt_algo、key_type 等其他 TDE 参数",
                "C. 改用 GRANT 命令为索引单独授权 TDE 访问权限",
                "D. 将 enable_tde=on 替换为 key_type 参数单独使用",
            ],
            "correct_answer": "B",
        },
        {
            "scenario": "Availability",
            "question_type": "MultipleChoice",
            "question": (
                "target_session_attrs 设置为 'prefer-standby' 时，"
                "若集群中所有备机均不可用，客户端的连接行为是？"
            ),
            "options": [
                "A. 直接报错，拒绝建立任何连接",
                "B. 自动回退，连接到主机",
                "C. 持续重试，直到有备机恢复",
                "D. 以只读模式强制连接到主机",
            ],
            "correct_answer": "B",
        },
    ]

    def __init__(
        self,
        count: int = 10,
        existing_questions: list | None = None,
        ensure_objective_questions: bool = True,
    ):
        self.count = count
        self.existing_questions = existing_questions or []
        self.ensure_objective_questions = ensure_objective_questions

    @staticmethod
    def _objective_count_bounds(count: int, ensure: bool) -> tuple[int, int]:
        """FillBlank + MultipleChoice 合计条数的 [min, max]（含端点）。"""
        if count <= 0:
            return 0, 0
        cap = min(count, max(0, int(round(count * 0.30))))
        if not ensure:
            return 0, cap
        if count <= 2 or cap < 1:
            return 0, cap
        return 1, cap

    def build_prompt(
        self,
        current_tag: str,
        tag_path: str = "",
        context: str = "",
        allowed_types: list | None = None,
    ) -> str:
        """
        构建问题蒸馏生成的提示词

        Args:
            current_tag: 当前标签/主题
            tag_path: 标签完整链路（可选）
            context: 参考上下文内容（可选）
            allowed_types: 本批可用的题型列表（来自文本信号分析）；None 表示不限制

        Returns:
            格式化后的提示词
        """
        # 处理已有问题
        existing_questions_section = ""
        if self.existing_questions:
            questions_list = "\n".join(f"- {q}" for q in self.existing_questions)
            existing_questions_section = f"""
## Existing Questions (Avoid generating similar ones):
{questions_list}
"""

        # 题型约束段
        if allowed_types:
            allowed_types_str = "、".join(allowed_types)
            allowed_types_section = (
                f"\n        - **本批题型约束（文本信号分析结果）**：本批仅允许使用以下题型："
                f"{allowed_types_str}。其余题型禁止出现，违反即视为无效题目。"
            )
        else:
            allowed_types_section = ""

        obj_min, obj_max = self._objective_count_bounds(self.count, self.ensure_objective_questions)
        if obj_max < obj_min:
            obj_max = obj_min
        if obj_min > 0:
            objective_quota_section = (
                f"\n        - **客观题条数（FillBlank + MultipleChoice 合计，硬约束）**："
                f"本批须输出 {self.count} 条题目，其中客观题合计须满足 **{obj_min} ≤ 条数 ≤ {obj_max}**。"
                f"当 {obj_min} ≥ 1 时，至少生成 1 条 FillBlank 或 1 条 MultipleChoice（可同时有两种，但合计不超过 {obj_max}），"
                f"且必须能在 Reference Context 中找到唯一可验证依据；禁止因图省事全部为主观题。"
            )
        else:
            objective_quota_section = (
                f"\n        - **客观题条数**：FillBlank + MultipleChoice 合计 **不超过 {obj_max}** 条；"
                f"count≤2 时可不出客观题。"
            )

        # 构建种子问题展示（few-shot 引导）
        # 为控制 prompt 长度，每类题型至多保留 1 条，问题正文截断至 80 字，
        # FillBlank/MultipleChoice 保留完整答案字段以示格式
        _MAX_Q_LEN = 80        # 问题文本最大显示字符数（超出截断+省略号）
        seen_types: set = set()
        seed_lines = []
        for ex in self._SEED_EXAMPLES:
            q_type = ex["question_type"]
            if q_type in seen_types:
                continue          # 同类型只展示第一条
            seen_types.add(q_type)
            q_text = ex["question"]
            if len(q_text) > _MAX_Q_LEN:
                q_text = q_text[:_MAX_Q_LEN] + "…"
            line = f'    {{"scenario": "{ex["scenario"]}", "question_type": "{q_type}", "question": "{q_text}"'
            if q_type == "FillBlank":
                blank = ex["blank_answer"]
                line += f', "blank_answer": "{blank}"'
            elif q_type == "MultipleChoice":
                opts_json = ", ".join(f'"{o}"' for o in ex["options"])
                line += f', "options": [{opts_json}], "correct_answer": "{ex["correct_answer"]}"'
            line += "}"
            seed_lines.append(line)
        seed_examples_str = ",\n".join(seed_lines)

        base_prompt = f"""
        # Role: 领域问题蒸馏专家

        ## Profile:
        - Description: 你是一个专业的知识问题生成助手，精通{current_tag}领域的知识。
        - Task: 为标签"{current_tag}"生成{self.count}个高质量、多样化的问题。
        - Context: 标签完整链路是：{tag_path}

        ## Reference Context:
        以下是与"{current_tag}"相关的参考资料，请基于这些内容生成问题：
        {context}

        ## Skills:
        1. 深入理解领域知识，能够识别和提取核心概念与关键知识点
        2. 设计多样化的问题类型，覆盖不同难度和认知层次（含填空题、单选题等客观题）
        3. 确保问题的准确性、清晰性和专业性
        4. 避免重复或高度相似的问题，保证问题集的多样性

        ## Seed Question Examples（格式与风格参考，内容必须完全来自上方 Reference Context）:
        【重要提示】以下示例仅展示「问题结构、措辞风格、题型格式」，
        其中涉及的参数名（如 walwriteraux_bind_cpu）、错误码（如 GAUSS-81700）、
        视图名（如 pg_stat_activity）等具体内容与当前 Reference Context 无关，
        禁止将示例中的任何实体、数值、路径照搬进最终输出。
        每一条生成的问题必须且只能使用上方 Reference Context 中出现的内容。
        [
{seed_examples_str}
        ]

        ## Workflow:
        1. 仔细阅读参考上下文，识别其中可被运维化的现象、参数、命令、配置、故障模式、具体数值
        2. 识别文本信号：错误码（GAUSS-/ALM-）→ 优先 Diagnostic/FillBlank；参数表（含数值/范围）→ FillBlank/ConfigExample；并列选项/策略 → MultipleChoice/RootCause；操作步骤 → Procedural；对比概念 → RootCause
        3. 在「运维过程场景轴」与「题型轴」上各选定 1 个（见 Constraints 第 4、5 条），并选定 1 个主标签用于自检{allowed_types_section}
        4. 按占比契约（Constraints 第 6 条）平衡场景与题型分布，核心排障/变更类必须过半
        5. 逐题复核：是否命中 4 条实用性硬要求、是否落入负面样本清单、是否与已有问题重复；客观题额外检查答案是否唯一可验证
        6. 输出最终的问题集，仅保留通过自检的问题，格式符合要求

        ## Constraints:
        1. 问题实用性硬要求（以下 4 条必须全部满足，任一违反即视为低质量问题，不得生成）：
        - 必须来自真实的运维 / 故障 / 调优 / 迁移 / 升级等生产场景，不能是纯文档概念背诵
        - 必须从用户视角提问（例如"我遇到 X 现象 / 我需要做 Y / 线上 Z 报错"），禁止文档作者视角（例如"本节介绍…"、"什么是 X"、"X 有哪些功能"）
        - 问题本身不得嵌入答案关键词、命令、参数值或结论（不能把标准答案泄露在题干里）；FillBlank 题除外——挖空处可以是"___"占位
        - 必须有明确的回答边界与可验证的目标，禁止"请介绍 / 请总结 / 有哪些 / 如何评价"这种开口式、无落点问法
        - 禁止「应重点确认哪两类/哪几项」这类**索引型空问题**：除非参考上下文中已**显式列出**可数的类别或条目名称，否则不得生成

        2. 上下文相关性：
        - 生成的问题必须能够从参考上下文中找到答案或相关信息
        - 问题应该覆盖上下文中的关键知识点
        - 不要生成上下文中没有涉及的问题

        3. 问题主题相关性：
        - 生成的问题必须与"{current_tag}"主题紧密相关
        - 确保全面覆盖该主题的核心知识点和关键概念

        4. 轴 A · 运维过程场景（每题必须先选定 1 类，整批至少覆盖 4 类不同场景）：
        - Incident（事件）：线上服务中断、宕机、连接失败、慢骤增等一线告警场景
        - Problem（问题）：反复发生的类故障、慢性症状、需要根因分析的长期问题
        - Change（变更）：升级、打补丁、参数调整、结构变更、迁移等计划性变动
        - Configuration（配置/CMDB）：参数、初始化项、实例属性、路径与资源清单的维护
        - Release（发布）：新版本、脚本化部署、对象发布、蓝绿/灰度推送
        - Capacity（容量）：存储、表空间、连接数、IO/CPU 容量评估与扩缩
        - Availability（可用性/HA）：主备、集群、故障切换、读写分离、RPO/RTO 达成
        - Continuity（连续性/灾备）：备份恢复、异地容灾、闪回、演练与恢复验证
        - Security（安全/合规）：权限、加密、审计、最小权限、合规标准落地
        - DevOpsSRE（自动化/可观测）：CI/CD、IaC、SLI/SLO、自动化运维与自愈
        - **单一类型 chunk 豁免（优先于下方强制下限，冲突时本条优先）**：
          在套用覆盖下限之前，请先自评 Reference Context 属于哪种类型；
          若属于以下任一类型，场景覆盖下限降为 **3 类**，Incident / Problem 等告警类允许为 0 条：
          (a) 纯配置/参数说明：文本主要由参数名称、取值范围、默认值、生效方式构成，不含故障案例；
          (b) 纯概念/背景介绍：文本只提供定义、架构原理或功能概述，不含操作步骤；
          (c) 单一操作流程：文本仅描述某一个操作（如安装、初始化、单次备份），无多场景分支。
          判断原则：**宁可放宽场景覆盖，也不得为凑类型生成与 context 无关的问题**（与 Constraint 2 上下文相关性保持一致）。
        - 强制下限（context 内容丰富时适用）：本批中 Incident / Problem / Change / Availability 四类合计占比 ≥ 50%
        - 当 {self.count} ≥ 3 时，至少覆盖 4 个不同场景轴（单一类型 chunk 豁免时降为 3 类）；count 较小时按比例放宽但不得只集中在 1 个场景

        5. 轴 B · 十三类题型（每题必须先选定 1 类，并遵守该题型的运维向合格问法）：
        - Factoid（运维事实型）：必须绑定现象/决策点/版本约束，问"X 在本场景下影响什么 / 与哪类故障相关 / 在当前版本默认值是否仍适用"；禁止裸的"什么是 X / X 的定义"
        - Diagnostic（诊断型）：给出现象 + 日志/指标/告警线索，问可能原因或"先查什么、看哪个视图/指标"
        - Procedural（操作步骤型）："如何在 {{环境约束}} 下完成 {{变更/操作}}"，必须带可验收目标（例如切换完成标志、备份成功标志）
        - RootCause（根因分析型）：针对反复或类故障模式，问机制层根因（锁/调度/资源/日志机制）与验证思路
        - BestPractice（最佳实践型）：题干必须含显式约束（维护窗口、RPO/RTO、预算、版本、数据量级），禁止无边界的"有哪些最佳实践"
        - ConfigExample（配置示例型）：问"依据本场景应修改哪些参数/配置项，前后如何检查"，题干不得写死目标参数值或结论
        - ScriptCommand（脚本/命令型）：问"给出可复用的脚本/命令骨架以完成 X"，敏感信息一律用占位符
        - FaultReproFix（故障复现与修复）：必须使用"在测试/预发/演练环境中"等限定词，禁止鼓励生产环境破坏性动作
        - ComplianceSecurity（合规/安全咨询）：围绕权限、加密、审计、最小权限、合规标准，必须与上下文中的安全相关段落绑定
        - DifferenceComparison（差异对比型）：聚焦两个或多个方案/模式/版本/配置的**核心区别**，题干需明确列出被对比对象（如"A 与 B 相比……"），答案要能给出可用于选型或决策的差异结论；禁止"请比较 X 和 Y 的所有区别"等无边界对比
        - VersionFeature（版本特性型）：围绕特定版本（或版本区间）新增/变更/废弃的特性、行为或限制，题干必须写明版本约束（如"在 X.Y 版本中……"），答案必须与原文版本范围严格对应；禁止跨版本泛化回答
        - FillBlank（完形/填空型）：从 context 中抽取唯一可验证的关键事实（参数值/取值范围/错误码含义/操作结果/限制条件），将其中的核心信息以"___"挖空；题干需提供足够定向线索；禁止挖空通用词汇（如"数据库"、"服务器"）；答案必须在 context 中有明确文字依据且唯一；output 必须包含 "blank_answer" 字段（多处挖空以"；"分隔）
        - MultipleChoice（单选题）：基于 context 中一个具体事实或规则，设计 1 个正确选项和 3 个与主题相关但事实有误的干扰项；干扰项须合理（不能明显错误到无需查原文即可排除）；禁止"以上都对"/"以上都不对"等万能选项；正确答案在 context 中必须有明确文字支持；output 必须包含 "options"（4 元素列表，格式 "A. 选项文本"）和 "correct_answer"（"A"/"B"/"C"/"D"）字段

        6. 占比契约（按 {self.count} 条总量自然语言化计算，出题前先核对）：
        - 核心排障/变更（Diagnostic + Procedural + RootCause）合计占比 ≥ 50%；当 {self.count} ≥ 3 时三者各至少 1 条；**若 chunk 命中 Constraint 4 中的「单一类型豁免」（纯配置/概念/单流程），此比例下限降为 ≥ 30%，且 Diagnostic/RootCause 各可为 0，但 Procedural 或 ConfigExample 至少 1 条**
        - 事实与配置（Factoid + ConfigExample）合计 ≤ 35%；Factoid 单独 ≤ 20%（防止定义题回潮）
        - BestPractice ≤ 25%，且每一条必须含显式约束从句
        - FaultReproFix ≤ 15%（count 很小时允许上限为 1），且一律测试/演练语境
        - ScriptCommand：若参考上下文中出现代码块、命令行、参数表、RMAN/SQL/shell 片段，则 ≥ 1 条；否则不强制
        - ComplianceSecurity：若参考上下文中出现"加密/权限/审计/合规/认证/TDE/SSL/角色"等关键词，则 ≥ 1 条；否则可为 0
        - DifferenceComparison：仅当 context 中**明确存在可对比的两种或以上方案/模式/版本**时可用，上限 ≤ 20%；否则不得强行凑对比题
        - VersionFeature：仅当 context 中**有明确版本号或版本区间**时可用，上限 ≤ 20%；否则不得强行凑版本题
        - FillBlank + MultipleChoice 的合计上限仍须遵守占比精神：在不超过上方「客观题条数」硬约束的前提下，FillBlank 单独 ≤ 20%（相对 {self.count} 条）、MultipleChoice 单独 ≤ 15%；二者每条须有 context 唯一可验证答案，无明确数值/条件可挖空/区分则不得勉强编造{objective_quota_section}
        - 同一场景轴 + 同一题型组合不得出现 3 条以上，避免类型坍缩

        7. 负面样本清单（任意命中即重写或丢弃，不得输出）：
        - 模糊开口问法：如"请介绍一下 X 的主要功能"、"X 有哪些注意事项"、"X 的优势是什么"
        - 文档视角问法：如"本节主要讲了什么"、"什么是 X"、"X 是如何定义的"、"请概述 X"
        - 答案嵌入问法：如"为什么把 innodb_buffer_pool_size 设置为物理内存的 70% 是合理的"（已把答案值写进题干）
        - 纯定义 Factoid：如"什么是闪回恢复区 / 什么是控制文件 / 什么是 ARCHIVELOG 模式"这类脱离现象与决策点的定义题；Factoid 必须改写为现象/决策/版本约束绑定形式
        - 生产破坏性引导：FaultReproFix 若未带"测试/预发/演练/实验环境"限定词，一律视为不合格
        - 无约束最佳实践：BestPractice 若未带维护窗口 / RPO-RTO / 版本 / 数据规模等任一约束，一律不合格
        - 无唯一答案填空：FillBlank 若挖空的内容在 context 中存在多个合理答案，或答案为通用词汇，一律不合格
        - 万能选项：MultipleChoice 中出现"以上都对"、"以上都不对"、"A 和 C 都对"等排列组合选项，一律不合格

        8. 问题质量要求：
        - 避免模糊或过于宽泛的表述
        - 避免可以简单用"是/否"回答的封闭性问题（FillBlank/MultipleChoice 除外）
        - 避免包含误导性假设的问题
        - 避免重复或高度相似的问题

        9. 溯源锚定（chunk 文本封闭性，最高优先级约束）：
        - 问题所有限定条件（实体名称 / 参数 / 文件名 / 字段名 / 数值 / 版本号 / 场景现象）**必须全部出现在上方"Reference Context"中**；不得引入 chunk 未提及的任何新事实或外部知识
        - 禁止在问题里写入**占位式具体值示例**（如 `event="vote_timeout" AND node_id="0x1234"`），任何带引号 / 反引号的具体值都必须是从上方 context 中原文摘录的
        - 禁止替换 context 中提到的路径 / 文件 / 命令为其他路径 / 文件 / 命令；题干只允许保留或删除 context 已有的内容，不允许替换
        - FillBlank 题的 blank_answer 和 MultipleChoice 题的正确选项内容必须能从 context 原文中直接引用或推导，不得编造
        - **Seed Examples 隔离规则**：种子示例中的参数名、错误码、视图名、SQL 语句、IP 地址等所有具体实体，若未出现在上方 Reference Context 中，一律**不得**出现在最终输出的任何问题里；命中此规则的题目视为越界，直接丢弃
        - 自检规则：把生成的问题交给"只能看当前 Reference Context、不能查其他资料"的人，他能否从 context 文本中直接得出结论？若答案为"否"，则该问题视为越界，**必须丢弃或重写**，不得出现在最终输出中

        {existing_questions_section}
        ## Output Format:
        - 返回 JSON 数组格式，其中每个元素是一个对象，包含轴A、轴B的标签和生成的问题。
        - 所有题型必须包含以下字段：
          - "scenario"：该问题归属的轴A（运维过程场景）标签，例如 "Incident", "Change" 等。
          - "question_type"：该问题归属的轴B（题型）标签，例如 "Diagnostic", "FillBlank", "MultipleChoice" 等。
          - "question"：生成的问题文本。
        - FillBlank 题额外必须包含：
          - "blank_answer"：填空正确答案（字符串，多处挖空以"；"分隔）。
        - MultipleChoice 题额外必须包含：
          - "options"：4 个选项的列表，格式为 ["A. 选项文本", "B. 选项文本", "C. 选项文本", "D. 选项文本"]。
          - "correct_answer"：正确答案选项字母，值为 "A"、"B"、"C" 或 "D"。
        - 其他题型的 blank_answer / options / correct_answer 字段可省略。
        - 不包含额外解释或说明。
        - 格式示例：
        [
            {{
                "scenario": "Incident",
                "question_type": "Diagnostic",
                "question": "如果某节点磁盘IO带宽占用率超过95%..."
            }},
            {{
                "scenario": "Change",
                "question_type": "Procedural",
                "question": "如何在维护窗口期内升级XX组件..."
            }},
            {{
                "scenario": "Configuration",
                "question_type": "FillBlank",
                "question": "该参数类型为___，属于___类参数，修改后须___生效。",
                "blank_answer": "整型；POSTMASTER；重启数据库"
            }},
            {{
                "scenario": "Incident",
                "question_type": "MultipleChoice",
                "question": "触发 GAUSS-XXXXX 报错后，正确的处理方式是？",
                "options": ["A. 操作1", "B. 操作2", "C. 操作3", "D. 操作4"],
                "correct_answer": "B"
            }}
        ]
        - 每个问题都必须是完整的、自包含的，无需依赖其他上下文即可理解和回答
        - 每条问题内部都须先完成「场景轴 + 题型 + 主标签」的自检，且整批 {self.count} 条问题必须整体满足 Constraints 第 4/5/6 条的双轴覆盖与占比契约（单一类型 chunk 豁免时按豁免后的下限执行）
        - 所有问题同时满足"问题实用性硬要求"的 4 条（含第 9 条溯源锚定），且任一命中负面样本清单的题不得出现在最终输出中
        - 禁止在问题里出现"步骤X"、"本节"、"原文"、"参考内容"等文档结构引用

        请开始生成问题："""

        return base_prompt


@PROMPT_REGISTRY.register()
class TextContentTypeAnalyzerPrompt(PromptABC):
    """
    # 用途：分析文本片段的信号特征，推断最适合从中出题的轴B题型列表
    # 应用场景：在 DistillQuestionGenerator 两步出题模式的第一步中使用
    #
    # 核心功能：
    # 检测文本中的结构信号（错误码、参数表、操作步骤、对比概念、安全内容等），
    # 输出 JSON 格式的文本信号列表和适合的题型列表，供后续出题约束使用
    #
    # 输出格式：
    # {"text_signals": [...], "suitable_types": [...]}
    # suitable_types 从轴B完整列表（13类）中选，最多5个，至少包含2个主观题型
    """

    def build_prompt(self, context: str) -> str:
        """
        构建文本内容题型分析的提示词

        Args:
            context: 待分析的文档片段

        Returns:
            格式化后的提示词
        """
        return f"""你是一名专业的运维知识问答题型分析师。请分析以下文档片段，判断哪些题型最适合从中出题。

## 待分析文档片段：
{context}

## 轴B题型完整列表（共13类）：
- Factoid：运维事实型（参数默认值/版本行为/现象关联）
- Diagnostic：诊断型（现象+日志线索→原因/排查路径）
- Procedural：操作步骤型（环境约束下的变更/操作流程）
- RootCause：根因分析型（机制层根因与验证思路）
- BestPractice：最佳实践型（带显式约束的优化建议）
- ConfigExample：配置示例型（参数修改场景与前后检查）
- ScriptCommand：脚本/命令型（可复用的命令骨架）
- FaultReproFix：故障复现与修复（测试/演练环境中）
- ComplianceSecurity：合规/安全咨询（权限/加密/审计）
- DifferenceComparison：差异对比型（两个或以上方案/模式/版本的核心区别，题干须明确被对比对象）
- VersionFeature：版本特性型（特定版本新增/变更/废弃的特性，题干须写明版本约束）
- FillBlank：完形/填空型（挖空参数值/错误码含义/操作结果）
- MultipleChoice：单选题（1个正确选项+3个合理干扰项）

## 信号→题型映射规则：
- 文本含 GAUSS-/ALM- 等错误码及 ACTION/CAUSE 字段 → 适合 Diagnostic、FillBlank
- 文本含具名参数 + 明确数值/取值范围/默认值 → 适合 FillBlank、ConfigExample、Factoid
- 文本含多个并列策略/模式/选项（3个以上） → 适合 MultipleChoice、RootCause
- 文本含操作步骤序列或命令行示例 → 适合 Procedural、ScriptCommand
- 文本含两个以上对比概念/机制差异或明确"A vs B"结构 → 适合 DifferenceComparison、RootCause
- 文本含明确版本号/版本区间及对应行为差异 → 适合 VersionFeature、Factoid
- 文本含权限/加密/审计/TDE/SSL/角色等安全关键词 → 适合 ComplianceSecurity
- 文本含监控指标/视图字段/查询语句 → 适合 Diagnostic、Factoid

## 输出要求：
1. 从文档片段中识别存在的信号类型，填入 text_signals 列表
2. 根据信号映射选出最适合的题型，填入 suitable_types 列表
3. suitable_types 最多 5 个，至少包含 2 个主观题型（Diagnostic/Procedural/RootCause 中至少 1 个）
4. **当 text_signals 含 error_code、parameter_with_value、enumerated_options、comparison_concepts 或 monitoring_metrics 中的任一项时，suitable_types 必须包含 FillBlank 或 MultipleChoice 中的至少一个**（二者可同在列表中，仍计为最多 5 个类型之一），除非全文极短且无表格/无数值/无错误码等可核验事实
5. 若除上述情况外文本信号不足以支撑客观题，可不选 FillBlank/MultipleChoice
6. 仅输出 JSON，不包含任何解释或额外文字

## 输出格式：
{{
    "text_signals": ["error_code", "parameter_with_value", ...],
    "suitable_types": ["Diagnostic", "FillBlank", "RootCause", ...]
}}

可选的 text_signals 枚举值：error_code（错误码）、parameter_with_value（参数+数值）、enumerated_options（并列选项/策略）、step_sequence（操作步骤）、comparison_concepts（对比概念）、security_keywords（安全关键词）、monitoring_metrics（监控指标/视图）、command_examples（命令示例）

请输出分析结果 JSON："""


@PROMPT_REGISTRY.register()
class EvolInstructPrompt(PromptABC):
    """
    # 用途：实现 Evol-Instruct 方法，将简单指令进化为更复杂的版本
    # 应用场景：在 EvolInstructGenerator 算子中使用
    #
    # 核心功能：
    # 1. 深度进化 (Depth Evolution)：
    #    - 添加约束 (Constraints)
    #    - 深化问题 (Deepen)
    #    - 具体化 (Concretizing)
    #    - 多步推理 (Reasoning)
    # 2. 广度进化 (Breadth Evolution)：
    #    - 创建同领域新问题
    #
    # 参数：
    # - evolution_type: 进化类型，可选 'depth' 或 'breadth' 或 'all'
    # - depth_strategies: 深度进化策略列表
    #
    # 特点：
    # - 随机选择进化策略增加多样性
    # - 进化后的指令更复杂、更具挑战性
    """

    # 深度进化基础 prompt（无上下文版本）
    DEPTH_BASE = """你是一个专业的问题改写专家。
你的目标是将给定的问题改写成更复杂的版本，使其更具挑战性。
但改写后的问题必须合理，且能够被人类理解和回答。
改写时不能省略#原始问题#中的非文本部分（如表格和代码），也不能省略#原始问题#中的输入内容。
你需要使用以下方法来增加问题的复杂度：
{method}
请尽量避免让#改写后的问题#变得冗长，#改写后的问题#只能比#原始问题#多10到20个字。
'#原始问题#'、'#改写后的问题#'、'原始问题'和'改写后的问题'这些标记不允许出现在#改写后的问题#中。
"""

    # 深度进化基础 prompt（带上下文版本）
    DEPTH_BASE_WITH_CONTEXT = """你是一个专业的问题改写专家。
你的目标是将给定的问题改写成更复杂的版本，使其更具挑战性。

【重要约束】
1. 改写后的问题必须严格基于#参考上下文#中的内容，不能脱离上下文范围
2. 改写后的问题的答案必须能够从#参考上下文#中找到或推断出来
3. 不能添加#参考上下文#中没有涉及的概念、技术或要求
4. 改写后的问题必须合理，且能够被人类理解和回答
5. 改写时不能省略#原始问题#中的非文本部分（如表格和代码）

你需要使用以下方法来增加问题的复杂度：
{method}

请尽量避免让#改写后的问题#变得冗长，#改写后的问题#只能比#原始问题#多10到20个字。
'#原始问题#'、'#改写后的问题#'、'#参考上下文#'等标记不允许出现在#改写后的问题#中。
"""

    # 广度进化基础 prompt（无上下文版本）
    BREADTH_BASE = """你是一个专业的问题创作专家。
你的目标是从#给定问题#中获取灵感，创作一个全新的问题。
这个新问题应该与#给定问题#属于同一领域，但更加独特和少见。
#创作的问题#的长度和复杂度应该与#给定问题#相似。
#创作的问题#必须合理，且能够被人类理解和回答。
'#给定问题#'、'#创作的问题#'、'给定问题'和'创作的问题'这些标记不允许出现在#创作的问题#中。
"""

    # 广度进化基础 prompt（带上下文版本）
    BREADTH_BASE_WITH_CONTEXT = """你是一个专业的问题创作专家。
你的目标是从#给定问题#中获取灵感，基于#参考上下文#创作一个全新的问题。

【重要约束】
1. 创作的问题必须严格基于#参考上下文#中的内容，不能脱离上下文范围
2. 创作的问题的答案必须能够从#参考上下文#中找到或推断出来
3. 不能添加#参考上下文#中没有涉及的概念、技术或场景
4. 这个新问题应该与#给定问题#属于同一领域，但角度或侧重点不同
5. #创作的问题#的长度和复杂度应该与#给定问题#相似
6. #创作的问题#必须合理，且能够被人类理解和回答

'#给定问题#'、'#创作的问题#'、'#参考上下文#'等标记不允许出现在#创作的问题#中。
"""

    # 深度进化策略
    DEPTH_STRATEGIES = {
        'constraints': "请在#原始问题#中添加一个或多个约束条件或要求（约束必须来自#参考上下文#中提到的内容）",
        'deepen': "如果#原始问题#涉及某些问题的询问，请增加询问的深度和广度（深度扩展必须基于#参考上下文#中的信息）",
        'concretizing': "请将#原始问题#中的通用概念替换为#参考上下文#中提到的更具体的概念",
        'reasoning': "如果#原始问题#可以通过简单的思考过程解决，请改写为明确要求多步推理的问题（推理步骤应基于#参考上下文#中的信息）"
    }

    # 深度进化策略（无上下文版本）
    DEPTH_STRATEGIES_NO_CONTEXT = {
        'constraints': "请在#原始问题#中添加一个或多个约束条件或要求",
        'deepen': "如果#原始问题#涉及某些问题的询问，请增加询问的深度和广度",
        'concretizing': "请将#原始问题#中的通用概念替换为更具体的概念",
        'reasoning': "如果#原始问题#可以通过简单的思考过程解决，请改写为明确要求多步推理的问题"
    }

    def __init__(
        self,
        evolution_type: str = 'all',
        depth_strategies: list = None,
        language: str = 'en',
        use_context: bool = False
    ):
        """
        初始化 EvolInstructPrompt

        Args:
            evolution_type: 进化类型，'depth'=仅深度进化, 'breadth'=仅广度进化, 'all'=随机选择
            depth_strategies: 深度进化策略列表，可选 ['constraints', 'deepen', 'concretizing', 'reasoning']
            language: 输出语言，'en'=英文, 'zh'=中文
            use_context: 是否使用上下文约束进化（启用后进化的问题会严格基于上下文）
        """
        self.evolution_type = evolution_type
        self.depth_strategies = depth_strategies or list(self.DEPTH_STRATEGIES.keys())
        self.language = language
        self.use_context = use_context

    def _build_depth_prompt(self, instruction: str, context: str = None, strategy: str = None) -> str:
        """构建深度进化 prompt"""
        if strategy is None:
            strategy = random.choice(self.depth_strategies)

        # 根据是否有上下文选择不同的模板和策略
        if self.use_context and context:
            method = self.DEPTH_STRATEGIES.get(strategy, self.DEPTH_STRATEGIES['constraints'])
            base = self.DEPTH_BASE_WITH_CONTEXT.format(method=method)
            prompt = base + f"\n#参考上下文#:\n{context}\n\n#原始问题#:\n{instruction}\n\n#改写后的问题#:\n"
        else:
            method = self.DEPTH_STRATEGIES_NO_CONTEXT.get(strategy, self.DEPTH_STRATEGIES_NO_CONTEXT['constraints'])
            base = self.DEPTH_BASE.format(method=method)
            prompt = base + f"\n#原始问题#:\n{instruction}\n#改写后的问题#:\n"

        return prompt

    def _build_breadth_prompt(self, instruction: str, context: str = None) -> str:
        """构建广度进化 prompt"""
        if self.use_context and context:
            prompt = self.BREADTH_BASE_WITH_CONTEXT + f"\n#参考上下文#:\n{context}\n\n#给定问题#:\n{instruction}\n\n#创作的问题#:\n"
        else:
            prompt = self.BREADTH_BASE + f"\n#给定问题#:\n{instruction}\n#创作的问题#:\n"
        return prompt

    def build_prompt(self, instruction: str, context: str = None, strategy: str = None) -> str:
        """
        构建进化 prompt

        Args:
            instruction: 原始指令
            context: 参考上下文（可选，启用 use_context 时用于约束进化范围）
            strategy: 指定策略，None 表示随机选择

        Returns:
            进化 prompt
        """
        if self.evolution_type == 'depth':
            return self._build_depth_prompt(instruction, context, strategy)
        elif self.evolution_type == 'breadth':
            return self._build_breadth_prompt(instruction, context)
        else:  # 'all'
            # 随机选择深度或广度进化
            if random.random() < 0.8:  # 80% 深度进化
                return self._build_depth_prompt(instruction, context, strategy)
            else:  # 20% 广度进化
                return self._build_breadth_prompt(instruction, context)

    def get_all_evolution_prompts(self, instruction: str, context: str = None) -> list:
        """
        获取所有进化策略的 prompts

        Args:
            instruction: 原始指令
            context: 参考上下文（可选）

        Returns:
            包含所有进化 prompts 的列表
        """
        prompts = []
        # 添加所有深度进化 prompts
        for strategy in self.depth_strategies:
            prompts.append({
                'type': 'depth',
                'strategy': strategy,
                'prompt': self._build_depth_prompt(instruction, context, strategy)
            })
        # 添加广度进化 prompt
        prompts.append({
            'type': 'breadth',
            'strategy': 'breadth',
            'prompt': self._build_breadth_prompt(instruction, context)
        })
        return prompts
