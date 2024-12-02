import os
import sys
import pandas as pd

# Get the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd()))
sys.path.insert(0, parent_dir)

from dataclasses import dataclass
from typing import List
from chatbot_api.providers.openai import OpenAIProvider
from chatbot_api.base import Role, Message

@dataclass
class Argument:
    main_point: str
    subpoints: List[str]

class ArgumentProcessor:
    def __init__(self, llm_provider: OpenAIProvider, batch_size):
        self.llm_provider = llm_provider
        self.batch_size = batch_size

    async def _create_initial_prompt(self, comments: List[str], topic: str) -> str:
        return f"""### Instructions:
                1. Analyze these comments on the topic: "{topic}"
                2. **Group similar arguments** into **main categories** (Main Arguments).
                3. Under each Main Argument, **list nuanced subpoints** or supporting details.
                4. Avoid duplication by consolidating similar points. 
                5. Use concise language and ensure each Main Argument stands alone clearly.

            
            ### More Details on the Task:
                I will provide you with a list of comments written in Finnish. However, your answer should be in English.
                Group logically similar arguments into main arguments and list nuanced subpoints under each. 
                Avoid logical duplication. Redundant arguments should be combined into a single, representative statement.
                When analysing whether a given argument is a main argument or a subpoint, consider the following:
                - Main arguments are broader and often encompass multiple subpoints. Main arguments can be altered 
                or combined if they are trying to convey the same idea from different perspectives, which can be divided into subpoints.
                - Subpoints are more specific and provide evidence or reasoning for the main argument.
                Format your response exactly like this example, including the exact markers. Do not include this prompt in your response.
                Notice that the main arguments are marked with "MAIN:" and the subpoints are marked with "SUB:".
                Note that there can be multiple subpoints under a main argument, and there can be multiple main arguments:
                
            ### Output Format:
                <ARGUMENTS>
                MAIN: [Main argument 1]
                SUB: [Subpoint 1.1]
                SUB: [Subpoint 1.2]
                MAIN: [Main argument 2]
                SUB: [Subpoint 2.1]
                </ARGUMENTS>

            ### Example Input:
                Vanhusten määrä suhteessa väestöön kasvaa koko ajan ja kaikilla tulee olla mahdollisuus arvokkaaseen loppuun.
                Resurssit eivät kertakaikkiaan riitä kaikille, siksi milestäni palvelukotipaikka taataan vain sellaisille jotka eivät pärjää kotonaan.
                He ovat ikänsä maksaneet veroja tähän maahan.
                Viime kädessä ongelma on henkilöresurssien saatavuus
                Tämä siis henkilön oma toive, eikä sukulaisten ja ympäristön toive. Kotihoito on kuitenkin monelle tärkeä siirtymä omavaraisuudesta hoivakotiin. Kotihoito pitää järjestää kaikille tarvitsijoille.
                Kunnan pitää edistää seniorikuntalaisten hyvinvointia erilaisilla palveluilla. Kotihoito, ateriapalvelu ja päivätoiminta ovat riittäviä tukimuotoja monelle, eikä kaikilla ole tarvetta jatkuvaan ympärivuorokautiseen tukeen ja apuun. Palvelukotipaikkoja tarvitaan tulevaisuudessa kuitenkin enemmän ja siihen on myös kunnan vastattava lisäämällä niin hoitohenkilökuntaa kuin hoitopaikkoja. Palveluasuminen pitää taata kaikille sitä tarvitseville.  En kuitenkaan usko, että kaikki vanhukset tulevat tarvitsemaan palvelukotipaikkaa.
                Yhteiskunnalla ei ole varaa subjektiiviseen oikeuteen palvelukotipaikkaan jokaiselle ikäihmiselle.
                Kotihoito on todella hyvä palvelumuoto silloin kun  vanhus vielä pärjää kotona. Mutta jokaisella vanhuksella tulisi olla oikeus palvelukotipaikkaan, jos hänestä siltä tuntuu, että ei kotona enää pärjää. Tarpeettoman pitkälle tätä siirtymää ei tule venyttää.
                Vanhukset ovat koko ikänsä veroja maksaneet niin heille pitäisi löytyä kunnasta myös palvelukotipaikka
                Kun olet tehnyt pitkän rupeaman elämässä, niin loppuelämä pitäisi olla helpompaa.
                Kasvavan ikääntyneiden ihmisten joukon hoitaminen edellyttää kotona asumisen tukemista ja kotihoitoon panostamista. Kuitenkin henkilön on lähtökohtaisesti päästävä muuhun hoitopaikkaan, jos asuminen kotona on mahdotonta.
                Vanhuksia ei saa jättää heitteille. 
                Kunnan on huolehdittava vanhuksistaan.
                Palvelukotipaikkoja on on oltava tarjolla riittävästi niitä tarvitseville ja haluaville vanhuksille. Aivan kaikki vanhukset eivät palvelukodeissa kuitenkaan pärjää tai niihin edes halua, joten myös vaihtoehtoja on oltava. Eli palvelukotipaikkojen lisäksi tulee varmistaa riittävät resurssit kotipalveluille, kehittää uusia viriketoiminnan ja yhteydenpidon muotoja (pian vanhuksetkin jo digiajassa!), tukea omaishoitajien jaksamista sekä tarjota laadukasta ja mahdollisimman kodinomaista laitoshoitoa sitä tarvitseville. 
                Kaikki vanhukset eivät varmasti halua palvelukotiin. Seniorikansalaisten arvokkaan vanhuuden paras paikka on kotona, johon tuotavia palveluita tulee kehittää ennakkoluulottomasti ja rohkeasti. Omaishoitajien työtä tulee tukea monin tavoin, esimerkiksi kaupungin omilla virkistysseteleillä. Sitten kun vanhuksen voimat eivät enää riitä kotona asumiseen, paikka löytyy hoivapalveluista. 
                On hyvä tavoite-kuitenkin varmaan mahdoton toteuttaa.Kotona asumisen palveluita paremmiksi.
                Ikääntyville ihmisille on tarjottava mahdollisuus arvokkaaseen elämään. He ovat kuitenkin omalla työllään osallistuneet yhteiskunnan ylläpitämiseen ja kasvuun
                Voiko muuten edes olla?
                Palvelukoti on parasta, kun omin voimin ei enää tule toimeen. 
                Pitää olla vapaus valita oman kunnon mukaan palvelut tai kotihoito. 
                Siinä vaiheessa kyllä, kun kotona ei enää yksin tai autettuna pärjää. 
                Tämä pitäisi kuulua perustarpeisiin. Liikaa luotetaan tänään omaisten tukeen vanhusten kotona asumiselle. Omaishoitajat tekevät kuitenkin tärkeää työtä, jopa oman jaksamisensa äärirajaan ja auttavat vanhuksiaan pysymään kotihoidossa nimellisellä korvauksella. Tämä onkin tärkeä apu kunnille vanhustenhoidossa.
                Silloin kyllä jos, ei kotona asuminen  enää onnistu. Asuminen oman kokomuksen mukaan on hinnakasta.
                Palvelukotipaikan tarpeellisuus on varmistettava.
                Tietysti kotonakin voi tuetusti asua.
                Kaikille vanhuksilla joilla on tarve palvelukotipaikalle, tulisi sellainen saada. Täytyy ottaa huomioon, että meillä on myös hyväkuntoisia kotonaan hyvin pärjääviä vanhuksia, joilla ei tällaista tarvetta ole ja silloin ei mielestäni heillä kuulu olla oikeus sitä saada. 
                Oma äitini on ollut hoiva-asumisessa ja nykyään palveluasumisessa, joten asia on minulle sikäli tuttu ja ajankohtainen. Vanhusten on päästävä palvelu- tai hoiva-asumisen piirin halutessaan ja silloin kun heillä on siihen tarve ja he eivät enää pärjää kotona. Kunnan oma palvelu tulee edullisemmaksi kuin yksityinen ja yksityisten toiminnassa on ollut viime vuosina erittäin vakavia puutteita.
                Niin kauan kuin se on vanhuksen oman tahdon, terveyden ja toimintakyvyn mukaan mahdollista.
                Kyllä, kun kotona ei enää voi turvallisesti asua, niin palvelukotipaikka pitää löytyä.
                Riittäisi alkuun, että taattaisiin edes niille, jotka palvelukotipaikkaa tarvitsevat. Nyt kotona hoidetaan kotihoidon turvin ihmisiä, jotka eivät oikeasti kotona enää pärjää. 
                Minusta meidän pitäisi sijoittaa ja mahdollistaa enemmän palvelukotipaikkoja.
                Ihmisille, jotka ovat ikänsä maksaneet veroja, pitää antaa mahdollisuus elämisen arvoiseen vanhuuteen. 
                Kannatan tätä ehdottomasti. Nykytrendi "kaikki hoidetaan kotona", ei toimi kaikkien ikääntyneiden kohdalla. Pitää olla mahdollista saada palvelukoti, paikka kun siihen on tarve. Yksinäinen muistisairas vanhus lukittuna omaan kotiin ei ole inhimillistä, hyvää vanhuutta. Hyvä, turvallinen riittävän tuen antava palvelukoti on tarjottava sitä tarvitseville.
                Tämä maa kykenee tarjoamaan asunnon, huonekalut siihen, lääkäripalvelut yksityisellä, tulkin, sähkölaskun, vesilaskun, puhelinlaskun ja päälle vielä käteistä rahaa loppuelämäksi kenelle tahansa, joka kykenee hyppäämään raja-aidan yli vaikka Turkissa, niin kyllä sen on kyettävä kustantamaan palvelukotipaikka ihmisille, jotka ovat maksaneet tänne veronsa koko työikänsä.
                Vanhusten palveluita tulee kehittää sekä parantaa kokonaisvaltaisesti. Kotihoidon tuki sekä mahdollisuudet pitää myös huomioida.
                Omaishoitajia ei tule unohtaa.
                Vanhusten omia mielipiteitä on kuunneltava AINA ensin!
                Ei kaikki vanhukset tahdo palvelukotipaikkaa vaan tahtovat asua kotona, se heille pitää suoda ja sitä tukea niin pitkään kuin se on järkevää, sitten kun on valmis palvelukotiin, tai kunto on niin huono että, ei pärjää kotona, niin on tarjottava palvelukotipaikaa. 
                Kyllä asia kuuluu ihmisarvoisen vanhuuden palveluihin.
                Vanhusten tulee saada viettää arvokas vanuhuus omassa kodissa niin kauan kun hän itse haluaa ja se on kohtuullisin toimin järjestettävissä. 
                Oikeus pitää olla yhtäläinen kaikille vanhuksille. Palvelumaksut tulee sovittaa maksukyvyn mukaisesti. 
                Jos vanhus tai vammainen on sellaisessa kunnossa, että hänen inhimillinen hoitonsa on mahdotonta kotona ja todennäköisesti se tulisi melko kalliiksi jo yhteiskunnallekin, täytyy jokaiselle  löytyä palvelukotipaikka. Vanhus haluaa useimmiten elää omassa kodissaan mahdollisimman kauan. 
                Laitoin täysin samaa mieltä, koska tämä on perusoikeus- ja ihmisarvokysymys. 
                Painopistettä tulee siirtää ehdottomasti kotihoidon puolelle huomioiden asiakkaan kunto, mutta hänen kuntonsa heiketessä on kaikille vanhuksille taattava oikeus palvelukotipaikkaan. Nykyään kuntapäättäjät päättävät mihin palveluasuntoon vanhus menee. Olisi hyvä, jos vanhus saisi omaistensa kanssa itse päättää palveluasumisestaan ja syömisestään. 
                Sosiaali- ja terveyspalvelujen tuleva uudistus tulee parantaa kunta-talouden vakautta ja
                ennustettavuutta. Uudistus tulee tehdä niin, että vanhus saa tarvitsemansa palvelun
                ja hoidon mahdollisimman läheltä.
                Kaikki vanhukset eivät edes halua palvelukotiin vaan pysyvät mieluummin kodeissaan niin kauan kuin mahdollista. Järkevästi suunniteltu kotihoito mahdollistaisi tämän. Tietysti sellaisten vanhusten on päästävä palvelutaloon, ketkä eivät pärjää enää kotioloissaan. 
                Kotkaan on rakennettu viimeaikoina ns. senioritaloja joissa voi asua kodinomaisissa olosuhteissa ja saada halutessan tarvittavat palvelut. Laitosmaiset asumismuodot ovat kalliita verrattuna omaishoitoon.
                Oikeus: kyllä. Pakote: ei.
                Silloin kun vanhuksen toimintakyky on selkeästi alentunut hänellä on edelleen oikeus hyvään elämään turvallisessa ympäristössä. Mielestäni kotihoidon laatuun tulisi panostaa, mikäli sitä tarjotaan ensisijaisena apukeinona.  
                Vanhuspalveluiden kysyntä kasvaa. Tarvitaan myös yksityisiä yrityksiä tuottamaan näitä palveluita. 
                Vanhusten palvelut tulee tuleivaisuudessa järjestää täysin uudella mallilla. Nykyjärjestelmää ei ole tulevaisuudessa mahdollista rahoittaa. Tämä on kuitenkin eduskunnan päätettävä asia.
                Kyllä. Jokaisella tulisi olla oikeus omannäköiseen ihmisarvoiseen elämään iästä riippumatta.
                P.S. Voitaisiin rakentaa myös tilavia yhteisöasumuksia, joka näyttää kodille, tuntuu kodille ja palvelun toteutuksessa olisi resurssit kohdillaan. 

            ### Desired Output on the Example Input:
                <ARGUMENTS> 
                    MAIN: [The current model for elderly care is not financially sustainable in the future.]
                    MAIN: [We will require more carehome spots in the future.]
                    SUB: [Need more private companies to make this possible.]

                    MAIN: [Something about how everyone should be given the opportunity to live their last years happy and/or with dignity / mental stimulation.]
                    SUB: [The last years of a person should be easier than before, because they deserve it.]
                    SUB: [You can’t just leave elderly people without care.]
                    SUB: [Impossible to envision a situation where elderly people are not supported.]
                    SUB: [We need more carehome spots.]
                    SUB: [The service needs to be available to everyone.]
                    SUB: [Payments for the care should be adapted to the person’s financial situation.]
                    SUB: [It is a basic human right.]

                    MAIN: [We don’t have enough resources / money to support every elderly person in a carehome, which is why the resources should only be reserved to those who simply can’t make it on their own.]
                    SUB: [Problem is specifically lack of employees.]
                    SUB: [Carehomes for those who actually need it / we need to check whether it is necessary so as to reserve resources.]
                    SUB: [It is sufficient to have spots for people who actually need it.]
                    SUB: [Currently there are people living at home even though they should have more support.]

                    MAIN: [Elderly people have paid taxes / paid their dues during their long life, so they should have the right to get help in their weaker years.]

                    MAIN: [Elderly people should decide for themselves whether they want to move to a carehome and have more power to decide where and how the arrangement is implemented.]
                    SUB: [The decision whether to move out of their own home to a carehome should be done based on the needs of the person.]
                    SUB: [Not everyone can handle living in a carehome.]
                    SUB: [Many people want to live at their own home for as long as they can.]

                    MAIN: [Home visits & other services (like food service, social activities) should be prioritized / are sufficient / the best option, when the elderly person can still live at home by themselves.]
                    SUB: [Reduce personal caregivers’ workload and mental strain with additional services from the city.]

                    MAIN: [Private carehomes have problems like high price, so cities should invest in their own services.]
                    SUB: [One option: community apartments for elderly people, which would be less resource-intensive for supplying care services.]
                </ARGUMENTS>

            ### Comments to analyze:
            {' '.join(comments)}"""
    
    async def _create_redundancy_prompt(self, argument: Argument) -> str:
        return f"""You are an assistant tasked with organizing arguments into a hierarchical structure. 
            Clearly separate main arguments from sub-arguments and merge redundant ideas into a single, representative statement.
            Reduce redundancy in the following argument structure:
            {self._format_argument(argument)}
            Identify and combine similar main arguments into broader, clearer categories.
            Preserve all unique subpoints under the most appropriate main argument.
            Use the exact same format as the input."""

    async def _create_consolidation_prompt(self, argument_structures: List[str]) -> str:
        return f"""Merge these argument structures into a single cohesive list.
            Combine similar main arguments into broader, clearer categories.
            Preserve all unique subpoints under the most appropriate main argument.
            Use the exact same format as the input:

            {' '.join(argument_structures)}"""

    async def _parse_argument_structure(self, text: str) -> List[Argument]:
        arguments = []
        current_main = None
        current_subs = []
        
        try:
            # Extract content between <ARGUMENTS> tags
            content = text.split("<ARGUMENTS>")[1].split("</ARGUMENTS>")[0].strip()
            
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("MAIN:"):
                    if current_main is not None:
                        arguments.append(Argument(current_main, current_subs))
                    current_main = line[5:].strip()
                    current_subs = []
                elif line.startswith("SUB:"):
                    current_subs.append(line[4:].strip())
            
            if current_main is not None:
                arguments.append(Argument(current_main, current_subs))
                
        except Exception as e:
            print(f"Error parsing argument structure: {e}")
            return []
            
        return arguments

    async def process_batch(self, comments: List[str], topic: str) -> List[Argument]:
        prompt = await self._create_initial_prompt(comments, topic)
        messages = [Message(Role("user"), prompt)]
        
        response = await self.llm_provider.generate(messages, temperature=0.3)
        return await self._parse_argument_structure(response.content)

    async def consolidate_arguments(self, argument_lists: List[List[Argument]]) -> List[Argument]:
        # Convert arguments to the format expected by the LLM
        structures = []
        for args in argument_lists:
            structure = "<ARGUMENTS>\n"
            for arg in args:
                structure += f"MAIN: {arg.main_point}\n"
                for sub in arg.subpoints:
                    structure += f"SUB: {sub}\n"
            structure += "</ARGUMENTS>"
            structures.append(structure)
        
        prompt = await self._create_consolidation_prompt(structures)
        messages = [{"role": "user", "content": prompt}]
        
        response = await self.llm_provider.generate(messages, temperature=0.3)
        return await self._parse_argument_structure(response.content)

    async def process_all_comments(self, comments: List[str], topic: str) -> List[Argument]:
        # Process comments in batches
        batches = [comments[i:i + self.batch_size] for i in range(0, len(comments), self.batch_size)]
        batch_results = []
        
        for batch in batches:
            result = await self.process_batch(batch, topic)
            batch_results.append(result)
            
        # Consolidate all batch results
        final_arguments = await self.consolidate_arguments(batch_results)
        return final_arguments
    
    def _format_argument(self, argument: Argument) -> str:
        formatted = f"MAIN: {argument.main_point}\n"
        for sub in argument.subpoints:
            formatted += f"SUB: {sub}\n"
        return formatted
    
    async def format_arguments(self, arguments: List[Argument]) -> str:
        formatted = ""
        for arg in arguments:
            formatted += self._format_argument(arg) + "\n"
        return formatted
    

# Example usage
async def main():
    # config
    api_key = os.getenv("OPENAI_API_KEY")
    openai_provider = OpenAIProvider(api_key, model="gpt-4o-2024-11-20")
    data_source_path = os.path.join(parent_dir, 'data', 'sources', 'kuntavaalit2021.csv')
    output_path = os.path.join(parent_dir, 'condensation', 'results', 'carehome-4o.txt')
    n_comments = 100
    batch_size = 100 
    topic = "All elderly people must be guaranteed the right to a place in a care home."

    # get comments
    df = pd.read_csv(data_source_path)
    comments = df['q8.explanation_fi'].dropna()[:n_comments].tolist()

    # process arguments
    processor = ArgumentProcessor(openai_provider, batch_size)
    arguments = await processor.process_batch(comments,topic)
    formatted_args = await processor.format_arguments(arguments)

    # save the output to a file
    with open(output_path, 'w') as f:
        f.write(formatted_args)

# run the main function
import asyncio
asyncio.run(main())


