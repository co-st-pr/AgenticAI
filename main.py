import time
from dotenv import load_dotenv
load_dotenv(verbose=True)


from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langsmith import traceable
#from langsmith.wrappers import wrap_openai



@traceable()
def main():
    print("Hello from langchain-course!")
    information = """The 2026 FIFA World Cup is the current and 23rd edition of the FIFA World Cup, the quadrennial international men's soccer championship contested by the national teams of the member associations of FIFA. The tournament began on June 11 and is scheduled to conclude on July 19, 2026.[1] It is jointly hosted by sixteen cities: eleven in the United States, three in Mexico, and two in Canada. The tournament is the first FIFA World Cup to be hosted by three nations and the first to include 48 teams, an expansion from the previous 32-team format.

The United 2026 bid beat a rival bid by Morocco during a final vote at the 68th FIFA Congress in Moscow. It is the first World Cup since 2002 to be co-hosted by multiple nations. With its past hosting of the 1970 and 1986 tournaments, Mexico became the first country to host or co-host the World Cup three times. The United States previously hosted the World Cup in 1994. By contrast, it is Canada's first time hosting or co-hosting the tournament. The event is returning to its traditional Northern Hemisphere summer schedule after the 2022 World Cup in Qatar was uniquely held in November and December.
 
As the host nations, Canada, Mexico, and the United States all automatically qualified. Cape Verde, Curaçao, Jordan, and Uzbekistan will all make their World Cup debuts. Argentina is the defending champion, having won its third World Cup in 2022.

Preparations for the tournament have drawn controversy, particularly over the United States's immigration and visa policies affecting qualified teams and their fans, Iran's participation during amid an ongoing war waged by the United States and Israel, and FIFA's use of dynamic ticket pricing.

The first goal of the 2026 FIFA World Cup was scored by Julián Quiñones of Mexico against South Africa in the 9th minute of the opening match of the tournament, held on June 11, 2026, at the Estadio Azteca in Mexico City.
    """

    summary_template = """
    give the information about {information} a topic I want you to summarize:
    1: primary details
    2: useful information"""

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOllama(temperature=0, model="gemma3:270m")
    #llm = ChatOpenAI(temperature=0, model="gpt-5")
    chain = summary_prompt_template | llm

    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
