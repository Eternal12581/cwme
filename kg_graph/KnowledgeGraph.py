# """
# Description: 
#     A class for constructing knowledge graphs from text using NLP and LLM models.

#     This class leverages Stanford CoreNLP for coreference resolution and named entity recognition, 
#     while utilizing large language models (LLM) for relation extraction and entity linking. 
#     Coreference resolution helps resolve pronouns and referring expressions to their corresponding entities. 
#     Entities are extracted with Stanford CoreNLP's named entity recognition, and relationships between entities 
#     are inferred using custom LLM prompts. The class optionally performs coreference resolution before entity 
#     and relation extraction to improve knowledge graph construction accuracy.
# """


# import stanza
# # stanza.download('en')
# from utils import *
# from pyvis.network import Network
# import asyncio
# from collections import defaultdict
# from psycopg2.extras import RealDictCursor
# from collections import deque 


# import re
# import sys
# import ast
# import kg_graph.schema as sm
# from time import time
# from pprint import pprint

# class KnowledgeGraph:
#     def __init__(self, 
#                  agent,
#                  prompts_path='kg_graph/conv_prompt',
#                  config=None, 
#                  model='Qwen/Qwen2.5-7B-Instruct',
#                  ner_llm=True):
#         """
#         A KnowledgeGraph object is responsible for constructing a knowledge graph from textual data, utilizing NLP and LLM models.
        
#         Attributes:
#             agent (CompHuSimAgent): The agent owning this KnowledgeGraph instance.
#             prompts_path (str): Path to the directory containing prompts for LLM queries.
#             config (dict): Configuration settings for the LLM model.
#             model (str): Identifier for the LLM model used for relation extraction and other NLP tasks.
#             ner (Stanza Pipeline object, optional): NLP pipeline for Named Entity Recognition. Defaults to None.
#             coref (Stanza Pipeline object): NLP pipeline for Coreference Resolution.
#         """
        
#         #Setup Connection Pool for Agent. Exit if fail.
#         self.connection_pool = agent.connection_pool
#         self.logger = agent.logger

#         if not self.connection_pool:
#             print("Failed DB Connection. Exiting")
#             self.logger.error(f"FAILED STARTUP: Error connecting to database. SHUTDOWN.")
#             exit()

#         if not agent: 
#             print('KnowledgeGraph has to belong to an agent, please provide CompHuSimAgent')
#             exit()
        
#         self.conn = self.connection_pool.getconn()
#         self.cur = self.conn.cursor(cursor_factory=RealDictCursor)
        
#         #owner
#         self.agent = agent
        
#         #NLP model
#         self.ner = None
#         if not ner_llm:
#             self.ner = stanza.Pipeline('en', processors='tokenize, ner')
#         self.coref = stanza.Pipeline('en', processors='tokenize, coref')
#         self.lemma = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma')

#         #prompt groups
#         self.token_sent = 0
#         self.token_received = 0
#         self.octo_token = 0
#         self.data_summary_prompt = open(f'{prompts_path}/data_summary_prompt.txt').read()
#         self.relational_prompt = open(f'{prompts_path}/prompt_edge_infer.txt').read()
#         self.entities_extract_prompt = open(f'{prompts_path}/entities_extraction_prompt.txt').read()
#         self.query_prompt = open(f'{prompts_path}/query_prompt.txt').read()
#         self.types_search = open(f'{prompts_path}/types_search.txt').read()
#         self.action_gen = open(f'{prompts_path}/action_generator.txt').read()
#         self.state_gen = open(f'{prompts_path}/next_state_generator.txt').read()
#         self.prompts_path = prompts_path
        
        
#         #kg_log
#         self.kg_log = ""
        
#         self.config = config
#         self.model = model

#     def reload_prompt(self):
#         """
#         Reloads the text prompts from the specified directory path into the KnowledgeGraph instance.
        
#         This method ensures that any updates to the prompt files are reflected in the KnowledgeGraph operations.
#         """
#         self.data_summary_prompt = open(f'{self.prompts_path}/data_summary_prompt.txt').read()
#         self.relational_prompt = open(f'{self.prompts_path}/prompt_edge_infer.txt').read()
#         self.entities_extract_prompt = open(f'{self.prompts_path}/entities_extraction_prompt.txt').read()
#         self.query_prompt = open(f'{self.prompts_path}/query_prompt.txt').read()
#         self.types_search = open(f'{self.prompts_path}/types_search.txt').read()
#         self.action_gen = open(f'{self.prompts_path}/action_generator.txt').read()
#         self.state_gen = open(f'{self.prompts_path}/next_state_generator.txt').read()

#     def reset(self):
#         self.token_sent = 0
#         self.token_received = 0
#         self.octo_token = 0
#         self.reload_prompt()
#         self.connection_pool = self.agent.connection_pool
#         self.logger = self.agent.logger

#         self.kg_log = ""


#     def coref_resolve(self, text):
#         """
#         Performs coreference resolution on the given text, replacing pronouns and referring expressions with the entities they refer to.
        
#         Parameters:
#             text (str): The text for which coreference resolution is to be performed.
        
#         Returns:
#             str: The text with resolved coreferences.
#         """
#         doc = self.coref(text)
#         resolved_doc = ""
#         for sentence in doc.sentences:
#             is_resolving = False
#             for token in sentence.words:
#                 if len(token.coref_chains) != 0 or is_resolving:
#                     longest_representative = None
#                     for chain in token.coref_chains:
#                         current_representative = chain.to_json()
#                         if longest_representative is None or len(current_representative['representative_text']) > len(longest_representative['representative_text']):
#                             longest_representative = current_representative
#                     representative = longest_representative
                    
#                     if representative:
#                         if 'is_start' in representative and representative['is_start']:
#                             is_resolving = True
#                             resolved_doc += " " + representative['representative_text']
#                         if 'is_end' in representative and representative['is_end']:
#                             is_resolving = False
                        
#                 elif not is_resolving:
#                     resolved_doc += " " + token.text
#         return resolved_doc


#     def lemmatize(self, text):
#         """
#         Performs lemmatization on the given text, converting words to their base forms.

#         Parameters:
#             text (str): The text to be lemmatized.

#         Returns:
#             str: The lemmatized text.
#         """
#         doc = self.lemma(text)
#         lemmatized_text = " ".join([word.lemma for sent in doc.sentences for word in sent.words])
#         return lemmatized_text
    

#     def extract_entities(self, text, coref_resolve=False, model='Qwen/Qwen2.5-7B-Instruct'):
#         """
#         Extracts named entities from the given text, optionally performing coreference resolution first.
        
#         Parameters:
#             text (str): The text from which entities are to be extracted.
#             coref_resolve (bool): Whether to perform coreference resolution before entity extraction. Defaults to False.
        
#         Returns:
#             list: A list of extracted entities.
#             str: The processed text, potentially with resolved coreferences.
#         """
#         if not coref_resolve:
#             text = self.coref_resolve(text)
#         text = self.lemmatize(text)
        
#         try:
#             if not self.ner:
#                 prompt = self.entities_extract_prompt.format(text)
#                 entities, token_count = retry(10, get_response, sm.Entities, model, prompt, json=True, config=self.config)
#                 # entities = get_response(model, prompt, json=True, config=self.config)
#             else: 
#                 entities = self.ner(text).ents
#             return entities, text
#         except Exception as e:
#             return {}, text

#     async def relations_extraction(self, text, entities=None, coref_resolve=False, store=True, iterations=1):
#         """
#         Extracts relations between entities from the text using LLM, potentially across multiple iterations for enhanced detail.
        
#         Parameters:
#             text (str): The text from which relations are to be extracted.
#             entities (list, optional): A list of entities to consider for relation extraction. Extracted from text if not provided.
#             coref_resolve (bool): Whether to perform coreference resolution before relation extraction. Defaults to False.
#             store (bool): Whether to store extracted relations in the database. Defaults to True.
#             json (bool): Whether to return the relations as JSON objects. Defaults to False.
#             iterations (int): Number of iterations for relation extraction to capture more details. Defaults to 3.
        
#         Returns:
#             list: A list of extracted relations in the format (entity1, relation, entity2).
#             str: The processed text, potentially with resolved coreferences.
#         """
#         try: 
#             # start_time = time()
#             agent_uuid = self.agent.get_uuid()
            
#             best_triples = defaultdict(lambda: ('', '', ''), {})
            
#             if coref_resolve:
#                 text = self.coref_resolve(text)

#             # print(f'finished coref resolve in : {time() - start_time}')
#             # print('resolved text: ', text)
#             # start_time = time()
            
            
#             if not entities:
#                 entities, text = self.extract_entities(text, coref_resolve=True, model=self.model)

#             # print(entities)
#             # print(f'finished NER in {time() - start_time}')
#             # print(entities)
#             # start_time = time()

#             tasks = []
#             prompt = self.relational_prompt.format(text, entities)
#             if iterations == 1:
#                 responses = [retry(10, get_response, sm.Relations, self.model, prompt, json=True, config=self.config)]
#                 # responses = [get_response(self.model, prompt, json=True, config=self.config)]
#             else:
#                 for _ in range(iterations):
#                     tasks.append(self.get_async_gpt_response(prompt, model=self.model, response_type='json_object'))
#                 responses = await asyncio.gather(*tasks)
            
#             # print(f'finshed rel extraction in {time() - start_time}')
#             # print(responses[0][0])
#             # start_time = time()
            
#             for res in responses:
#                 # print(res)
#                 res, token_count = res
#                 self.token_sent += token_count['sent']
#                 self.token_received += token_count['received']
#                 triples = res['rels']
                
#                 for triple in triples:
#                     if len(triple) == 3:
#                         #only process valid trips
#                         key = (triple[0], triple[2])  # (entity1, entity2) as the key
#                         if len(triple[1]) > len(best_triples[key][1]):  # Check if the new relation is more detailed
#                             best_triples[key] = triple
            
#             all_triples = list(best_triples.values())

#             # print(f'filter and cleaning rels in {time() - start_time}')
#             # print(all_triples)
#             # start_time = time()

#             if store:
#                 self.store_knowledge_graph(all_triples, entities, agent_uuid)


#             # print(f'finshed storing in {time() - start_time}')
#             # start_time = time()
            
#             return res, text
#         except Exception as e:
#             self.logger.error('ERROR SAVING: ', e)
#             print(f'error saving {e}')
#             return

#     async def get_async_gpt_response(self, gpt_prompt, model='Qwen/Qwen2.5-7B-Instruct',response_type='text'):
#         """
#         Generate a response using local LLM and run it in a thread for async workflows.

#         Parameters:
#             config (ConfigParser): The configuration parser object that contains the API key.
#             gpt_prompt (str): The prompt to pass to API
#             model (str): Local model identifier (HuggingFace model id or configured local alias).

#         Returns:
#             str: The content of the response generated by API

#         Note:
#             - The temperature parameter controls the randomness of the output. A higher value makes the output more random, while a lower value makes it more deterministic.
#             - The max_tokens parameter controls the maximum length of the generated response.
#             - The frequency_penalty parameter can be used to reduce the likelihood of frequent words/phrases.
#         """
#         wants_json = (response_type == 'json_object')
#         content, token_count = await asyncio.to_thread(
#             get_response,
#             model,
#             gpt_prompt,
#             wants_json,
#             None,
#             self.config
#         )
#         return content, token_count

#     def store_knowledge_graph(self, fact_tuples, ner_res, agent_uuid=None):
#         """
#         Stores the extracted knowledge graph information in the database, ensuring that entities are only stored
#         if they are part of relations that are successfully inserted into the database.
        
#         Parameters:
#             fact_tuples (list of tuples): Extracted relations to be stored.
#             ner_res (list): Named entity recognition results.
#             agent_uuid (str, optional): UUID of the agent owning the knowledge graph. Uses the instance's agent if not provided.
#         """
#         try: 
#             if not agent_uuid:
#                 agent_uuid = self.agent.get_uuid()
            
#             # Set up database connection and cursor
#             with self.connection_pool.getconn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
#                 entities = {e[0]: e[1] for e in ner_res['entities']}
#                 entity_ids = {}
#                 relation_ids = {}

#                 # First check and insert relations, then process entities if relations exist
#                 new_facts = []
#                 for subject, relation, obj in fact_tuples:
#                     if subject in entities and obj in entities:
#                         new_facts.append((subject, relation, obj))
                        
#                 # Insert relations and check existing ones
#                 for _, relation, _ in new_facts:
#                     if relation not in relation_ids:
#                         relation_ids[relation] = self._insert_relation(cur, relation)

#                 # Process entities only if their relations are being inserted
#                 for subject, relation, obj in new_facts:
#                     if relation_ids[relation]:  # Only process entities if the relation was processed
#                         if subject not in entity_ids:
#                             entity_ids[subject] = self._insert_entity(cur, subject, entities[subject])
#                         if obj not in entity_ids:
#                             entity_ids[obj] = self._insert_entity(cur, obj, entities[obj])

#                         # Prepare for insertion into fact_tuples
#                         subject_id = entity_ids[subject]
#                         relation_id = relation_ids[relation]
#                         object_id = entity_ids[obj]
#                         self._insert_fact_tuple(cur, subject_id, relation_id, object_id, agent_uuid)
#             self.connection_pool.putconn(conn)
#             cur.close()
            
#         except Exception as e:
#             self.logger.error('ERROR STORING: ', e)
#             print(f"Error storing knowledge graph: {e}")

#     def _insert_entity(self, cur, name, etype):
#         """
#         Inserts an entity into the database if it doesn't already exist.
#         """
#         cur.execute("SELECT id FROM entities WHERE name = %s;", (name,))
#         result = cur.fetchone()
#         if result:
#             return result['id']
#         else:
#             cur.execute("INSERT INTO entities (name, type) VALUES (%s, %s) RETURNING id;", (name, etype,))
#             return cur.fetchone()['id']

#     def _insert_relation(self, cur, name):
#         """
#         Inserts a relation into the database if it doesn't already exist.
#         """
#         cur.execute("SELECT id FROM relationships WHERE relationship_type = %s;", (name,))
#         result = cur.fetchone()
#         if result:
#             return result['id']
#         else:
#             cur.execute("INSERT INTO relationships (relationship_type) VALUES (%s) RETURNING id;", (name,))
#             return cur.fetchone()['id']

#     def _insert_fact_tuple(self, cur, source_id, relation_id, target_id, agent_uuid):
#         """
#         Inserts a fact tuple into the database if it doesn't already exist.
#         """
#         cur.execute("""
#             SELECT EXISTS (
#                 SELECT 1 FROM fact_tuples
#                 WHERE source_entity_id = %s AND relationship_id = %s AND target_entity_id = %s AND agent_uuid = %s
#             );
#         """, (source_id, relation_id, target_id, agent_uuid))
#         exists = cur.fetchone()['exists']
#         if not exists:
#             cur.execute("""
#                 INSERT INTO fact_tuples (source_entity_id, relationship_id, target_entity_id, agent_uuid) 
#                 VALUES (%s, %s, %s, %s);
#             """, (source_id, relation_id, target_id, agent_uuid))

#     def get_all_entities(self, type=None):
#         """
#         Retrieves all entities from the database that are part of a fact tuple for the current agent, 
#         optionally filtered by type.
        
#         Parameters:
#             type (tuple, optional): Types of entities to retrieve. Defaults to None.
        
#         Returns:
#             dict: A mapping of entity names to their IDs.
#         """
#         cur = None
#         conn = None
#         try:
#             conn = self.connection_pool.getconn()
#             cur = conn.cursor(cursor_factory=RealDictCursor)
            
#             base_query = """
#             SELECT e.id, e.name 
#             FROM entities e
#             JOIN fact_tuples ft ON e.id = ft.source_entity_id OR e.id = ft.target_entity_id
#             WHERE ft.agent_uuid = %s
#             """

#             if type:
#                 cur.execute(base_query + "AND e.type IN %s;", (self.agent.get_uuid(), type))
#             else:
#                 cur.execute(base_query, (self.agent.get_uuid(),))

#             entities = cur.fetchall()
#             return {entity['name']: entity['id'] for entity in entities}
#         finally:
#             if cur is not None:
#                 cur.close()
#             if conn is not None:
#                 self.connection_pool.putconn(conn)


#     def get_related_relations(self, entity_ids, agent_uuid=None):
#         """
#         Retrieves relations related to the specified entities from the database.
        
#         Parameters:
#             entity_ids (list of int): IDs of entities whose relations are to be retrieved.
#             agent_uuid (str, optional): UUID of the agent owning the knowledge graph. Defaults to None.
        
#         Returns:
#             list: A list of dictionaries representing the related relations.
#         """
#         # Initialize variables for connection and cursor to None
#         conn = None
#         cur = None

#         try:
#             # Ensure connection is set up
#             conn = self.connection_pool.getconn()
#             cur = conn.cursor(cursor_factory=RealDictCursor)
            
#             if not agent_uuid:
#                 agent_uuid = self.agent.get_uuid()
            
#             # Convert the list of entity IDs to a tuple for SQL query
#             entity_ids_tuple = tuple(entity_ids)
            
#             # SQL query
#             query =  """
#                 SELECT ft.*, e1.name AS source_entity_name, e2.name AS target_entity_name, r.relationship_type
#                 FROM fact_tuples ft
#                 JOIN entities e1 ON ft.source_entity_id = e1.id
#                 JOIN relationships r ON ft.relationship_id = r.id
#                 JOIN entities e2 ON ft.target_entity_id = e2.id
#                 WHERE agent_uuid = %s
#                 AND (source_entity_id IN %s OR target_entity_id IN %s);
#                 """
            
#             # Execute the query with parameters
#             cur.execute(query, (agent_uuid, entity_ids_tuple, entity_ids_tuple))
            
#             # Fetch and return the results
#             return cur.fetchall()
#         finally:
#             # Ensure resources are always cleaned up properly
#             if cur is not None:
#                 cur.close()
#             if conn is not None:
#                 self.connection_pool.putconn(conn)    

#     def bfs_path(self, start, end):
#         """
#         Finds a path between two entities using Breadth-First Search (BFS).
        
#         Parameters:
#             start (int): The starting entity ID.
#             end (int): The target entity ID.
        
#         Returns:
#             list: The path of entity IDs from start to end, if exists.
#         """
#         agent_uuid = self.agent.get_uuid()
#         conn = self.connection_pool.getconn()
#         cur = conn.cursor(cursor_factory=RealDictCursor)
#         query = """
#             SELECT ft.target_entity_id AS neighbor_id
#             FROM fact_tuples ft
#             WHERE ft.agent_uuid = %s AND ft.source_entity_id = %s
#             UNION
#             SELECT ft.source_entity_id AS neighbor_id
#             FROM fact_tuples ft
#             WHERE ft.agent_uuid = %s AND ft.target_entity_id = %s;
#             """
        
#         visited = {start: None}  # Maps each node to its predecessor in the path
#         queue = deque([start])
#         try: 
#             while queue:
#                 current = queue.pop()
#                 if current == end:
#                     break  # Exit if the end entity is found

#                 # Fetch neighbors from the graph
#                 cur.execute(query, (agent_uuid, current, agent_uuid, current))
#                 neighbors = [row['neighbor_id'] for row in cur.fetchall()]
#                 for neighbor in neighbors:
#                     if neighbor not in visited:
#                         visited[neighbor] = current  # Map the neighbor back to the current node
#                         queue.append(neighbor)

#             self.connection_pool.putconn(conn)
#             cur.close()
            
#             # Reconstruct the path from end to start (if exists)
#             path = []
#             while end is not None:
#                 path.append(end)
#                 end = visited[end]
#             return path[::-1]  # Return reversed path
#         except Exception as e:
#             return [start,end]

#     def get_k_hop_neighbors(self, path_entities, agent_uuid, k):
#         """
#         Retrieves k-hop neighbors for entities along a path.
        
#         Parameters:
#             path_entities (list of int): Entity IDs forming a path.
#             agent_uuid (str): UUID of the agent owning the knowledge graph.
#             k (int): The number of hops to consider.
        
#         Returns:
#             list: A list of relations within k hops of the path entities.
#         """
#         conn = self.connection_pool.getconn()
#         cur = conn.cursor(cursor_factory=RealDictCursor)
        
#         all_relations = {}
#         entities_to_explore = set(path_entities)
        
#         for _ in range(k):
#             if not entities_to_explore:
#                 break
            
#             entities_tuple = tuple(entities_to_explore)
#             query_relations = """
#                 SELECT ft.*, e1.name AS source_entity_name, e2.name AS target_entity_name, 
#                 r.relationship_type, ft.created_at
#                 FROM fact_tuples ft
#                 JOIN entities e1 ON ft.source_entity_id = e1.id
#                 JOIN relationships r ON ft.relationship_id = r.id
#                 JOIN entities e2 ON ft.target_entity_id = e2.id
#                 WHERE ft.agent_uuid = %s
#                 AND (ft.source_entity_id IN %s OR ft.target_entity_id IN %s);
#                 """
            
#             cur.execute(query_relations, (agent_uuid, entities_tuple, entities_tuple))
#             results = cur.fetchall()
#             for trip in results:
#                 all_relations[trip['id']] = trip
            
#             # Update entities to explore for the next hop
#             new_entities = {r['source_entity_id'] for r in results}.union({r['target_entity_id'] for r in results})
#             entities_to_explore = new_entities - set(path_entities)
#             path_entities.extend(entities_to_explore)
        
#         self.connection_pool.putconn(conn)
#         cur.close()
#         return list(all_relations.values())

#     def get_temporal_k_hop(self, entity_ids, agent_uuid=None, k=3):
#         """
#         Retrieves a temporally ordered list of k-hop neighbors for specified entities.
        
#         Parameters:
#             entity_ids (list of int): The IDs of entities to explore.
#             agent_uuid (str, optional): UUID of the agent owning the knowledge graph. Defaults to None.
#             k (int): The number of hops to consider. Defaults to 3.
        
#         Returns:
#             list: A list of relations within k hops of the specified entities, ordered by time.
#         """
#         if not agent_uuid:
#             agent_uuid = self.agent.get_uuid()
        
#         if len(entity_ids) < 2:
#             return []  # Need at least two entities to define a path
        
#         # Find a static path between the two entities
#         path_entities = self.bfs_path(*entity_ids)
#         # print(path_entities)
#         # Get k-hop neighbors from the path
#         all_relations = self.get_k_hop_neighbors(path_entities, agent_uuid, k)
        
#         # Sort all relations by time in increasing order
#         all_relations.sort(key=lambda x: x['created_at'])
        
#         return all_relations

#     def _validate_entity_types(self, output):
#         """
#         Validates the selected entity types based on the query and given instructions.

#         Parameters:
#         - query (str): The query string.
#         - output (dict): The selected entity types in the format {index: "type"}.

#         Returns:
#         - bool: True if the output is valid, False otherwise.
#         """
#         # Define the allowed entity types
#         allowed_types = ["OBJ", "LOC", "PER"]

#         # Ensure output is a dictionary with integer keys and string values
#         if not isinstance(output, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in output.items()):
#             print("Output should be a dictionary with integer keys and string values.")
#             return False

#         # Ensure all values are among the allowed entity types
#         if not all(v in allowed_types for v in output.values()):
#             print("Output contains invalid entity types.")
#             return False

#         # If all checks pass, the output is valid
#         return True

#     def query(self, query, additional_context="", asker_uuid=None, summarize=False):
#         """
#         Queries the knowledge graph and optionally summarizes the results.
        
#         Parameters:
#             query (str): The query string.
#             asker_uuid (str, optional): UUID of the agent making the query. Defaults to None.
#             summarize (bool): Whether to summarize the query results. Defaults to False.
        
#         Returns:
#             The query results, potentially summarized.
#         """
#         try:
#             if not asker_uuid:
#                 asker_uuid = self.agent.get_uuid()
            
#             all_types = ['OBJ', 'LOC', 'PER']

            
#             #get related types to filter out entity list
#             # types = get_response(self.model, self.types_search.format(query), json=True, config=self.config)
#             # print(types)
#             types, token_count = retry(5, get_response, self._validate_entity_types , self.model, self.types_search.format(query), json=True, config=self.config)
#             self.token_sent += token_count['sent']
#             self.token_received += token_count['received']

#             types_tuple = tuple([all_types[int(i)] for i in types.keys()])
#             filtered_entities = self.get_all_entities(types_tuple)
            
#             # results = retry(10, get_response, lambda a: isinstance((a), list), self.model, self.query_prompt.format(query, filtered_entities), config=self.config)
            
#             results, token_count = get_response(self.model, self.query_prompt.format(query, filtered_entities), config=self.config)
#             self.token_sent += token_count['sent']
#             self.token_received += token_count['received']
            
#             # print(results)
#             self.octo_token += token_count['sent'] + token_count['received']
            
#             # print(results)
#             match = re.search(r"\[(.*?)\]", results)
            
#             if match:
#                 results = f"[{match.group(1)}]"
#             try:
#                 entities_list = ast.literal_eval(results)
#                 related_entites = self.get_temporal_k_hop([filtered_entities[entity] for entity in entities_list], k=2)
#             except Exception as e:
#                 print(e)
#                 related_entites = []
            
#             format = '%H:%M:%S'
#             timestamps = ""
#             for trip in related_entites:
#                 t = trip['created_at'].strftime(format)
#                 timestamps += f'{t}: {trip["source_entity_name"]} -- {trip["relationship_type"]} --> {trip["target_entity_name"]}\n'
            
#             # print(timestamps)
            
#             # return timestamps
            
#             res, token_count = get_response(self.model, self.data_summary_prompt.format(additional_context + "\n GRAPH: \n" + timestamps, query), config=self.config)
#             self.token_sent += token_count['sent']
#             self.token_received += token_count['received']
            
#             self.octo_token += token_count['sent'] + token_count['received']
            
#             return res 

#         except Exception as e:
#             self.logger.error('ERROR QUERYING: ', e)
#             print(f'error querying {e}')
        
#     def visualize(self, relations):
#         """
#         Visualizes the extracted relations as a knowledge graph.
        
#         Parameters:
#             relations (list of tuples): The relations to visualize.
#         """
#         # Create a PyVis network with directed edges
#         net = Network(notebook=True, directed=True)

#         added_entities = set()
        
#         # Add nodes and edges
#         for source, relation, target in relations:
#             if source not in added_entities:
#                 net.add_node(source, label=source)
#                 added_entities.add(source)
#             if target not in added_entities:
#                 net.add_node(target, label=target)
#                 added_entities.add(target)
                
#             # Add directed edge with a title
#             net.add_edge(source, target, title=relation, arrowStrikethrough=True)

#         # Generate and display the graph
#         net.show_buttons(filter_=True)
#         net.show("network.html")
             
#     def visualize_all(self, agent_uuid=None):
#         # Ensure connection and cursor are set up
#         conn = None
#         cur = None
#         try: 
#             if not conn:
#                 conn = self.connection_pool.getconn()
#             if not cur:
#                 cur = conn.cursor(cursor_factory=RealDictCursor)
#             if not agent_uuid:
#                 agent_uuid = self.agent.get_uuid()
            
#             # Include JOINs to fetch names instead of IDs
#             cur.execute("""
#                 SELECT e1.name AS source_entity_name, r.relationship_type, e2.name AS target_entity_name 
#                 FROM fact_tuples ft
#                 JOIN entities e1 ON ft.source_entity_id = e1.id
#                 JOIN entities e2 ON ft.target_entity_id = e2.id
#                 JOIN relationships r ON ft.relationship_id = r.id
#                 WHERE agent_uuid = %s;
#                 """, (agent_uuid,))
            
#             results = cur.fetchall()
            
#             relations = [(fact['source_entity_name'], fact['relationship_type'], fact['target_entity_name']) for fact in results]
#             self.visualize(relations)
#         finally:
#             if cur is not None:
#                 cur.close()
#             if conn is not None:
#                 self.connection_pool.putconn(conn)
                
#     #Planning, Algorithm 1
#     def get_trajectory(self, valid_objects, task, observation, inventory, MAX_STEPS, MAX_QUERIES, sequence=None, reflection="", model='Qwen/Qwen2.5-7B-Instruct'):
#         """
#         Generates a trajectory of a state-action-reward-state-action-reward (SARSA) sequences
#         for a given task utilizing knowledge from a knowledge graph.

#         Parameters:
#         - valid_actions (list): A list of actions that can be performed in the environment.
#                                 Each action should be applicable to one or more objects
#                                 within the valid_objects list.
#         - valid_objects (list): A list of objects that are present in the environment and
#                                 can be interacted with through the actions in valid_actions.
#         - task (str): A string describing the task to be achieved. The task should be
#                     interpretable based on the knowledge available in the knowledge graph,
#                     guiding the trajectory generation.
#         - observation (str): The initial observation

#         Returns:
#         - trajectory (list of tuples): A list where each element is a tuple representing a
#                                     SARSA sequence: (state, action, reward, next_state, next_action).
#                                     - state (dict): The current state of the environment, describing
#                                                     the status of various objects.
#                                     - action (str): The action taken in the current state.
#                                     - reward (float): The reward received after taking the action.
#                                     - next_state (dict): The state of the environment after the action
#                                                             is taken.
#                                     - next_action (str): The next action to be taken in the new state.
#         """
#         # Initialize the trajectory list and the initial state based on the observation
#         state_0 = {
#             'observation': observation,
#             'inventory': inventory,
#             'valid_receptacles': valid_objects,
#         }
        
#         env_sum = {
#             'task': task,
#             'reflection': reflection[-4:]
#         }
        
#         current_sars = {
#                 'state': state_0,
#                 'action': '<PREDICT>',
#                 'reward (env response)': '<PREDICT>',
#                 'next state': None,
#                 'done or termination': None
#             }
        
#         if not sequence:
#             sequence = [
#                 current_sars
#             ]
#         else:
#             sequence.append(current_sars)

#         done = False
#         step = 0
#         plan_printout = ""
        
#         while step < MAX_STEPS and not done:
#             current_sars = sequence[-1]
#             res = self.plan_next_action_reward(env_sum, sequence[-3:], model, max_query=MAX_QUERIES)
#             current_sars['action'] = res['predicted_action']
#             current_sars['reward (env response)'] = res['predicted_response']
#             current_sars['next state'] = '<PREDICT>'
#             current_sars['done or termination'] = '<PREDICT>'
            
#             sequence[-1] = current_sars
#             res = self.plan_next_state_termination(env_sum, sequence[-3:], model, max_query=MAX_QUERIES)
#             current_sars['next state'] = res['state']
#             current_sars['done or termination'] = res['task_completion']
#             done = False
            
#             sequence[-1] = current_sars
#             next_sars = {
#                 'state': res['state'],
#                 'action': '<PREDICT>',
#                 'reward (env response)': '<PREDICT>',
#                 'next state': None,
#                 'done or termination': None
#             }
            
#             plan_printout += f"\n<Action> {sequence[-1]['action']} - <Response>{sequence[-1]['reward (env response)']}"     
#             sequence.append(next_sars)
            
#             step += 1
            
#         return sequence, plan_printout, self.token_sent, self.token_received

#     def plan_next_action_reward(self, env_sum, sequence, model, max_query=3):
#         """
#         Plans the next action and predicts its reward, incorporating reflections on past attempts.
        
#         Parameters:
#             env_sum (dict): Summary of the current environment state and task.
#             sequence (list): The current sequence of SARSA tuples.
#             reflection (list of str, optional): Reflections on past unsuccessful actions. Defaults to an empty list.
#             max_query (int): The maximum number of KG queries allowed. Defaults to 5.
        
#         Returns:
#             dict: Predicted next action and its expected reward.
#         """
#         predicted = False
#         num_query = 0
#         kg_log = """"""
#         while not predicted:
#             action_gen_prompt = self.action_gen.format(env_sum, sequence, kg_log)
#             res, token_count = retry(10, get_response, sm.ActionPrediction, model, action_gen_prompt, json=True, schema=sm.ActionPrediction, config=self.config)
#             self.token_sent += token_count['sent']
#             self.token_received += token_count['received']
#             if res['query']:
#                 num_query += 1
#                 if num_query >= max_query:
#                     qa = f"""
# QUERY: {res['query']}
# RESPONSE: 'maximum number of query reached. Please pick an arbitrary action that is relevant to the task' 
#                     """
#                 else: 
#                     qa = f"""
# QUERY: {res['query']}
# RESPONSE: {self.query(res['query'])} 
#                     """
                    
#                 # print(qa)
#                 kg_log += qa
#             if res['predicted_action']:
#                 predicted = True
#                 # pprint(res)
#         return res
        
#     def plan_next_state_termination(self, env_sum, sequence, model, max_query=3):
#         """
#         Predicts the next state of the environment and whether the task is terminated.
        
#         Parameters:
#             env_sum (dict): Summary of the current environment state and task.
#             sequence (list): The current sequence of SARSA tuples.
#             max_query (int): The maximum number of KG queries allowed. Defaults to 5.
        
#         Returns:
#             dict: Predicted next state and task completion status.
#         """
#         print(model)
#         kg_log = """"""
#         predicted = False
#         num_query = 0
#         while not predicted:
#             state_gen_prompt = self.state_gen.format(sequence, kg_log)
#             res, token_count = retry(10, get_response, sm.StateWithQuery, model, state_gen_prompt, json=True, schema=sm.StateWithQuery, config=self.config)
#             self.token_sent += token_count['sent']
#             self.token_received += token_count['received']
#             if res['query']:
#                 num_query += 1
#                 if num_query >= max_query:
#                     qa = f"""
# QUERY: {res['query']}
# RESPONSE: 'maximum number of query reached. Please pick an arbitrary action that is relevant to the task' 
#                     """
#                 else: 
#                     qa = f"""
# QUERY: {res['query']}
# RESPONSE: {self.query(res['query'])} 
#                     """
                    
#                 # print(qa)
#                 kg_log += qa
#             if res['state']:
#                 predicted = True
#                 # pprint(res)
#         return res
    
    
#     def memory_reset(self):
#         """
#         Clear all memories in KG

#         Parameters:

#         Returns:
#             None
#         """
#         try:
#             uuid = self.agent.this_uuid
#             # Get a connection from the pool
#             conn = self.connection_pool.getconn()
#             cur = conn.cursor(cursor_factory=RealDictCursor)

#             # Disable foreign key checks (optional, depending on DB setup)
#             cur.execute("SET session_replication_role = 'replica';")

#             # Delete from fact_tuples table
#             cur.execute("""
#                 DELETE FROM fact_tuples
#                 WHERE agent_uuid = %s;
#             """, (uuid,))

#             # Find entities associated with the deleted fact tuples
#             cur.execute("""
#                 DELETE FROM entities
#                 WHERE id IN (
#                     SELECT e.id
#                     FROM entities e
#                     LEFT JOIN fact_tuples ft ON e.id = ft.source_entity_id OR e.id = ft.target_entity_id
#                     WHERE ft.agent_uuid = %s
#                     GROUP BY e.id
#                     HAVING COUNT(ft.id) = 0
#                 );
#             """, (uuid,))

#             # Clean up relationships if necessary
#             cur.execute("""
#                 DELETE FROM relationships
#                 WHERE id NOT IN (
#                     SELECT DISTINCT relationship_id FROM fact_tuples
#                 );
#             """)

#             # Re-enable foreign key checks
#             cur.execute("SET session_replication_role = 'origin';")

#             # Commit changes
#             conn.commit()

#         except Exception as e:
#             if conn:
#                 conn.rollback()
#             print(f"Error removing data by UUID: {e}")
#         finally:
#             if cur:
#                 cur.close()
#             if conn:
#                 self.connection_pool.putconn(conn)

import stanza
from utils import *
from utils import (
    planner_substitute_navigation,
    planner_should_replace_action,
    planner_phase_navigation_hint,
)
from pyvis.network import Network
import asyncio
from collections import defaultdict
from psycopg2.extras import RealDictCursor
from collections import deque 

import re
import sys
import ast
import kg_graph.schema as sm
from time import time
from pprint import pprint

class KnowledgeGraph:
    def __init__(self, 
                 agent,
                 prompts_path='kg_graph/conv_prompt',
                 config=None, 
                #  model='/mnt/nfsData19/Zhaoshuyuan/Houxinrui/model/Qwen3.5-9B',
                 model='/data/ZhaoShuyuan/Zhaoshuyuan/HouXinrui/Baseline/Qwen3.5-9B',
                 cognition_model=None,
                 ner_llm=True):
        """
        完全离线版 · 禁用stanza · 适配本地Qwen模型
        model: 感知/KG 抽取用小模型；cognition_model: 规划用大模型
        """
        # 数据库连接
        self.connection_pool = agent.connection_pool
        self.logger = agent.logger

        if not self.connection_pool:
            print("Failed DB Connection. Exiting")
            self.logger.error(f"FAILED STARTUP: Error connecting to database. SHUTDOWN.")
            exit()

        if not agent:
            print('KnowledgeGraph has to belong to an agent, please provide CompHuSimAgent')
            exit()
        
        self.conn = self.connection_pool.getconn()
        self.cur = self.conn.cursor(cursor_factory=RealDictCursor)
        
        # 所有者
        self.agent = agent
        
        # ======================
        # 🔥 完全禁用 stanza，防止联网下载
        # ======================
        self.ner = None
        self.coref = None   # 禁用
        self.lemma = None   # 禁用

        # 提示词
        self.token_sent = 0
        self.token_received = 0
        self.octo_token = 0
        self.data_summary_prompt = open(f'{prompts_path}/data_summary_prompt.txt').read()
        self.relational_prompt = open(f'{prompts_path}/prompt_edge_infer.txt').read()
        self.entities_extract_prompt = open(f'{prompts_path}/entities_extraction_prompt.txt').read()
        self.query_prompt = open(f'{prompts_path}/query_prompt.txt').read()
        self.types_search = open(f'{prompts_path}/types_search.txt').read()
        self.action_gen = open(f'{prompts_path}/action_generator.txt').read()
        self.state_gen = open(f'{prompts_path}/next_state_generator.txt').read()
        self.prompts_path = prompts_path
        
        # 日志
        self.kg_log = ""
        
        self.config = config
        self.model = model
        self.cognition_model = cognition_model or model

    def reload_prompt(self):
        self.data_summary_prompt = open(f'{self.prompts_path}/data_summary_prompt.txt').read()
        self.relational_prompt = open(f'{self.prompts_path}/prompt_edge_infer.txt').read()
        self.entities_extract_prompt = open(f'{self.prompts_path}/entities_extraction_prompt.txt').read()
        self.query_prompt = open(f'{self.prompts_path}/query_prompt.txt').read()
        self.types_search = open(f'{self.prompts_path}/types_search.txt').read()
        self.action_gen = open(f'{self.prompts_path}/action_generator.txt').read()
        self.state_gen = open(f'{self.prompts_path}/next_state_generator.txt').read()

    def reset(self):
        self.token_sent = 0
        self.token_received = 0
        self.octo_token = 0
        self.reload_prompt()
        self.connection_pool = self.agent.connection_pool
        self.logger = self.agent.logger
        self.kg_log = ""

    @property
    def router(self):
        return getattr(self.agent, "router", None)

    def _llm_retry(
        self,
        task: str,
        val_func,
        prompt: str,
        max_attempts: int = 10,
        json: bool = True,
        schema=None,
    ):
        router = self.router
        if router:
            return router.retry(
                max_attempts, task, val_func, prompt, json=json, schema=schema,
            )
        model = self.cognition_model if task.startswith("plan_") else self.model
        return retry(
            max_attempts,
            get_response,
            val_func,
            model,
            prompt,
            json=json,
            schema=schema,
            config=self.config,
        )

    def _llm_call(self, task: str, prompt: str, json: bool = False, schema=None):
        router = self.router
        if router:
            return router.call(task, prompt, json=json, schema=schema)
        model = self.cognition_model if task.startswith("plan_") else self.model
        return get_response(
            model, prompt, json=json, schema=schema, config=self.config,
        )

    def _planner_context_block(self) -> str:
        agent = self.agent
        if hasattr(agent, "get_planner_context"):
            try:
                return agent.get_planner_context() or ""
            except Exception:
                pass
        if hasattr(agent, "get_working_memory_context"):
            try:
                return agent.get_working_memory_context() or ""
            except Exception:
                pass
        return ""

    # ======================
    # 直接返回原文，不使用 stanza
    # ======================
    def coref_resolve(self, text):
        return text

    def lemmatize(self, text):
        return text

    # ======================
    # 完全使用 LLM 提取实体
    # ======================
    def extract_entities(self, text, coref_resolve=False, model=None):
        try:
            prompt = self.entities_extract_prompt.format(text)
            entities, token_count = self._llm_retry(
                "kg_ner", sm.Entities, prompt, max_attempts=10, json=True,
            )
            return entities, text
        except Exception as e:
            return {}, text

    async def relations_extraction(self, text, entities=None, coref_resolve=False, store=True, iterations=1):
        try: 
            agent_uuid = self.agent.get_uuid()
            best_triples = defaultdict(lambda: ('', '', ''), {})

            if not entities:
                entities, text = self.extract_entities(text, coref_resolve=True)

            tasks = []
            prompt = self.relational_prompt.format(text, entities)
            if iterations == 1:
                responses = [self._llm_retry(
                    "kg_relations", sm.Relations, prompt, max_attempts=10, json=True,
                )]
            else:
                for _ in range(iterations):
                    tasks.append(self.get_async_gpt_response(prompt, model=self.model, response_type='json_object'))
                responses = await asyncio.gather(*tasks)
            
            for res in responses:
                res, token_count = res
                self.token_sent += token_count['sent']
                self.token_received += token_count['received']
                triples = res['rels']
                
                for triple in triples:
                    if len(triple) == 3:
                        key = (triple[0], triple[2])
                        if len(triple[1]) > len(best_triples[key][1]):
                            best_triples[key] = triple
            
            all_triples = list(best_triples.values())

            if store:
                self.store_knowledge_graph(all_triples, entities, agent_uuid)

            return res, text
        except Exception as e:
            self.logger.error('ERROR SAVING: ', e)
            print(f'error saving {e}')
            return

    async def get_async_gpt_response(self, gpt_prompt, model=None, response_type='text'):
        wants_json = (response_type == 'json_object')
        router = self.router
        if router:
            content, token_count = await asyncio.to_thread(
                router.call,
                "kg_relations",
                gpt_prompt,
                wants_json,
                None,
            )
            return content, token_count
        content, token_count = await asyncio.to_thread(
            get_response,
            self.model,
            gpt_prompt,
            wants_json,
            None,
            self.config
        )
        return content, token_count

    # ======================
    # 以下所有函数完全不变
    # ======================
    def store_knowledge_graph(self, fact_tuples, ner_res, agent_uuid=None):
        try: 
            if not agent_uuid:
                agent_uuid = self.agent.get_uuid()
            
            with self.connection_pool.getconn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
                entities = {e[0]: e[1] for e in ner_res['entities']}
                entity_ids = {}
                relation_ids = {}

                new_facts = []
                for subject, relation, obj in fact_tuples:
                    if subject in entities and obj in entities:
                        new_facts.append((subject, relation, obj))
                        
                for _, relation, _ in new_facts:
                    if relation not in relation_ids:
                        relation_ids[relation] = self._insert_relation(cur, relation)

                for subject, relation, obj in new_facts:
                    if relation_ids[relation]:
                        if subject not in entity_ids:
                            entity_ids[subject] = self._insert_entity(cur, subject, entities[subject])
                        if obj not in entity_ids:
                            entity_ids[obj] = self._insert_entity(cur, obj, entities[obj])

                        subject_id = entity_ids[subject]
                        relation_id = relation_ids[relation]
                        object_id = entity_ids[obj]
                        self._insert_fact_tuple(cur, subject_id, relation_id, object_id, agent_uuid)
            self.connection_pool.putconn(conn)
            
        except Exception as e:
            self.logger.error('ERROR STORING: ', e)
            print(f"Error storing knowledge graph: {e}")

    def _insert_entity(self, cur, name, etype):
        cur.execute("SELECT id FROM entities WHERE name = %s;", (name,))
        result = cur.fetchone()
        if result:
            return result['id']
        else:
            cur.execute("INSERT INTO entities (name, type) VALUES (%s, %s) RETURNING id;", (name, etype,))
            return cur.fetchone()['id']

    def _insert_relation(self, cur, name):
        cur.execute("SELECT id FROM relationships WHERE relationship_type = %s;", (name,))
        result = cur.fetchone()
        if result:
            return result['id']
        else:
            cur.execute("INSERT INTO relationships (relationship_type) VALUES (%s) RETURNING id;", (name,))
            return cur.fetchone()['id']

    def _insert_fact_tuple(self, cur, source_id, relation_id, target_id, agent_uuid):
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM fact_tuples
                WHERE source_entity_id = %s AND relationship_id = %s AND target_entity_id = %s AND agent_uuid = %s
            );
        """, (source_id, relation_id, target_id, agent_uuid))
        exists = cur.fetchone()['exists']
        if not exists:
            cur.execute("""
                INSERT INTO fact_tuples (source_entity_id, relationship_id, target_entity_id, agent_uuid) 
                VALUES (%s, %s, %s, %s);
            """, (source_id, relation_id, target_id, agent_uuid))

    def get_all_entities(self, type=None):
        cur = None
        conn = None
        try:
            conn = self.connection_pool.getconn()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            base_query = """
            SELECT e.id, e.name 
            FROM entities e
            JOIN fact_tuples ft ON e.id = ft.source_entity_id OR e.id = ft.target_entity_id
            WHERE ft.agent_uuid = %s
            """

            if type:
                cur.execute(base_query + "AND e.type IN %s;", (self.agent.get_uuid(), type))
            else:
                cur.execute(base_query, (self.agent.get_uuid(),))

            entities = cur.fetchall()
            return {entity['name']: entity['id'] for entity in entities}
        finally:
            if cur:
                cur.close()
            if conn:
                self.connection_pool.putconn(conn)

    def _runtime_kg_query_enabled(self) -> bool:
        """Whether planner-side PostgreSQL KG queries are meaningful this run.

        Formal read-only experiments use the frozen HKG artifact and a fresh
        run-scoped PostgreSQL KG.  Querying that empty scoped KG still invokes
        three perception-model calls per query, although the HKG context has
        already been assembled for the planner.  Keep queries enabled for
        legacy/shared-KG runs and for experiments that explicitly store KG
        facts.
        """
        cfg = getattr(self.agent, "agent_config", None) or {}
        if not isinstance(cfg, dict):
            return True
        exp = cfg.get("EXPERIMENT") or {}
        if not isinstance(exp, dict):
            return True
        store_kg = exp.get("STORE_KG", True)
        run_scoped = exp.get("RUN_SCOPED_KG", True)
        return not (store_kg is False and run_scoped is not False)

    def get_related_relations(self, entity_ids, agent_uuid=None):
        conn = None
        cur = None
        try:
            conn = self.connection_pool.getconn()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            if not agent_uuid:
                agent_uuid = self.agent.get_uuid()
            
            entity_ids_tuple = tuple(entity_ids)
            
            query =  """
                SELECT ft.*, e1.name AS source_entity_name, e2.name AS target_entity_name, r.relationship_type
                FROM fact_tuples ft
                JOIN entities e1 ON ft.source_entity_id = e1.id
                JOIN relationships r ON ft.relationship_id = r.id
                JOIN entities e2 ON ft.target_entity_id = e2.id
                WHERE agent_uuid = %s
                AND (source_entity_id IN %s OR target_entity_id IN %s);
                """
            
            cur.execute(query, (agent_uuid, entity_ids_tuple, entity_ids_tuple))
            return cur.fetchall()
        finally:
            if cur:
                cur.close()
            if conn:
                self.connection_pool.putconn(conn)    

    def bfs_path(self, start, end):
        agent_uuid = self.agent.get_uuid()
        conn = self.connection_pool.getconn()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        query = """
            SELECT ft.target_entity_id AS neighbor_id
            FROM fact_tuples ft
            WHERE ft.agent_uuid = %s AND ft.source_entity_id = %s
            UNION
            SELECT ft.source_entity_id AS neighbor_id
            FROM fact_tuples ft
            WHERE ft.agent_uuid = %s AND ft.target_entity_id = %s;
            """
        
        visited = {start: None}
        queue = deque([start])
        try: 
            while queue:
                current = queue.pop()
                if current == end:
                    break 

                cur.execute(query, (agent_uuid, current, agent_uuid, current))
                neighbors = [row['neighbor_id'] for row in cur.fetchall()]
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited[neighbor] = current
                        queue.append(neighbor)

            self.connection_pool.putconn(conn)
            cur.close()
            
            path = []
            while end is not None:
                path.append(end)
                end = visited[end]
            return path[::-1]
        except Exception as e:
            return [start,end]

    def get_k_hop_neighbors(self, path_entities, agent_uuid, k):
        conn = self.connection_pool.getconn()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        all_relations = {}
        entities_to_explore = set(path_entities)
        
        for _ in range(k):
            if not entities_to_explore:
                break
            
            entities_tuple = tuple(entities_to_explore)
            query_relations = """
                SELECT ft.*, e1.name AS source_entity_name, e2.name AS target_entity_name, 
                r.relationship_type, ft.created_at
                FROM fact_tuples ft
                JOIN entities e1 ON ft.source_entity_id = e1.id
                JOIN relationships r ON ft.relationship_id = r.id
                JOIN entities e2 ON ft.target_entity_id = e2.id
                WHERE ft.agent_uuid = %s
                AND (ft.source_entity_id IN %s OR ft.target_entity_id IN %s);
                """
            
            cur.execute(query_relations, (agent_uuid, entities_tuple, entities_tuple))
            results = cur.fetchall()
            for trip in results:
                all_relations[trip['id']] = trip
            
            new_entities = {r['source_entity_id'] for r in results}.union({r['target_entity_id'] for r in results})
            entities_to_explore = new_entities - set(path_entities)
            path_entities.extend(entities_to_explore)
        
        self.connection_pool.putconn(conn)
        cur.close()
        return list(all_relations.values())

    def get_temporal_k_hop(self, entity_ids, agent_uuid=None, k=3):
        if not agent_uuid:
            agent_uuid = self.agent.get_uuid()
        
        if len(entity_ids) < 2:
            return [] 
        
        path_entities = self.bfs_path(*entity_ids)
        all_relations = self.get_k_hop_neighbors(path_entities, agent_uuid, k)
        
        all_relations.sort(key=lambda x: x['created_at'])
        
        return all_relations

    def _validate_entity_types(self, output):
        allowed_types = ["OBJ", "LOC", "PER"]

        if not isinstance(output, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in output.items()):
            return False

        if not all(v in allowed_types for v in output.values()):
            return False

        return True

    def query(self, query, additional_context="", asker_uuid=None, summarize=False):
        try:
            if not self._runtime_kg_query_enabled():
                if self.logger:
                    self.logger.info(
                        "[KG] skipped runtime query in read-only run-scoped protocol"
                    )
                return (
                    "No runtime PostgreSQL KG query was performed. Use the supplied "
                    "historical/world-model context and current observation."
                )
            if not asker_uuid:
                asker_uuid = self.agent.get_uuid()

            wm_ctx = ""
            if hasattr(self.agent, "get_memory_fusion_context"):
                try:
                    wm_ctx = self.agent.get_memory_fusion_context(query) or ""
                except Exception:
                    wm_ctx = ""
            if not wm_ctx and hasattr(self.agent, "get_working_memory_context"):
                try:
                    wm_ctx = self.agent.get_working_memory_context() or ""
                except Exception:
                    wm_ctx = ""
            if wm_ctx:
                additional_context = (wm_ctx + "\n\n" + (additional_context or "")).strip()
            
            all_types = ['OBJ', 'LOC', 'PER']

            types_prompt = (
                self.types_search.format(query)
                + "\n\nReturn ONLY a valid JSON object. All property names MUST be strings in double quotes."
                + " Do not include markdown fences or any extra text."
            )
            types, token_count = self._llm_retry(
                "kg_query_types",
                self._validate_entity_types,
                types_prompt,
                max_attempts=5,
                json=True,
            )
            self.token_sent += token_count['sent']
            self.token_received += token_count['received']

            types_tuple = tuple([all_types[int(i)] for i in types.keys()])
            filtered_entities = self.get_all_entities(types_tuple)
            
            results, token_count = self._llm_call(
                "kg_query_entities",
                self.query_prompt.format(query, filtered_entities),
            )
            self.token_sent += token_count['sent']
            self.token_received += token_count['received']
            
            self.octo_token += token_count['sent'] + token_count['received']
            
            match = re.search(r"\[(.*?)\]", results)
            
            if match:
                results = f"[{match.group(1)}]"
            try:
                entities_list = ast.literal_eval(results)
                related_entites = self.get_temporal_k_hop([filtered_entities[entity] for entity in entities_list], k=2)
            except Exception as e:
                related_entites = []
            
            format = '%H:%M:%S'
            timestamps = ""
            for trip in related_entites:
                t = trip['created_at'].strftime(format)
                timestamps += f'{t}: {trip["source_entity_name"]} -- {trip["relationship_type"]} --> {trip["target_entity_name"]}\n'
            
            res, token_count = self._llm_call(
                "kg_query_summary",
                self.data_summary_prompt.format(
                    additional_context + "\n GRAPH: \n" + timestamps, query,
                ),
            )
            self.token_sent += token_count['sent']
            self.token_received += token_count['received']
            
            self.octo_token += token_count['sent'] + token_count['received']
            
            return res 

        except Exception as e:
            # Use exception() to include traceback and avoid logging format errors.
            self.logger.exception("ERROR QUERYING")
            print(f'error querying {e}')
        
    def visualize(self, relations):
        net = Network(notebook=True, directed=True)
        added_entities = set()
        
        for source, relation, target in relations:
            if source not in added_entities:
                net.add_node(source, label=source)
                added_entities.add(source)
            if target not in added_entities:
                net.add_node(target, label=target)
                added_entities.add(target)
                
            net.add_edge(source, target, title=relation, arrowStrikethrough=True)

        net.show_buttons(filter_=True)
        net.show("network.html")
             
    def visualize_all(self, agent_uuid=None):
        conn = None
        cur = None
        try: 
            if not conn:
                conn = self.connection_pool.getconn()
            if not cur:
                cur = conn.cursor(cursor_factory=RealDictCursor)
            if not agent_uuid:
                agent_uuid = self.agent.get_uuid()
            
            cur.execute("""
                SELECT e1.name AS source_entity_name, r.relationship_type, e2.name AS target_entity_name 
                FROM fact_tuples ft
                JOIN entities e1 ON ft.source_entity_id = e1.id
                JOIN entities e2 ON ft.target_entity_id = e2.id
                JOIN relationships r ON ft.relationship_id = r.id
                WHERE agent_uuid = %s;
                """, (agent_uuid,))
            
            results = cur.fetchall()
            
            relations = [(fact['source_entity_name'], fact['relationship_type'], fact['target_entity_name']) for fact in results]
            self.visualize(relations)
        finally:
            if cur:
                cur.close()
            if conn:
                self.connection_pool.putconn(conn)
                
    def get_trajectory(self, valid_objects, task, observation, inventory, MAX_STEPS, MAX_QUERIES, sequence=None, reflection="", model=None):
        state_0 = {
            'observation': observation,
            'inventory': inventory,
            'valid_receptacles': valid_objects,
        }
        
        env_sum = {
            'task': task,
            'reflection': reflection[-4:]
        }
        
        current_sars = {
                'state': state_0,
                'action': '<PREDICT>',
                'reward (env response)': '<PREDICT>',
                'next state': None,
                'done or termination': None
            }
        
        if not sequence:
            sequence = [current_sars]
        else:
            sequence.append(current_sars)

        done = False
        step = 0
        plan_printout = ""
        plan_model = model or self.cognition_model
        
        while step < MAX_STEPS and not done:
            current_sars = sequence[-1]
            res = self.plan_next_action_reward(env_sum, sequence[-3:], plan_model, max_query=MAX_QUERIES)
            current_sars['action'] = res['predicted_action']
            current_sars['reward (env response)'] = res['predicted_response']
            current_sars['next state'] = '<PREDICT>'
            current_sars['done or termination'] = '<PREDICT>'
            
            sequence[-1] = current_sars
            if self._state_prediction_enabled():
                res = self.plan_next_state_termination(
                    env_sum, sequence[-3:], plan_model, max_query=MAX_QUERIES,
                )
            else:
                # The live execution path is the source of truth. Keep the
                # state stable while collecting action candidates and let
                # env.step refresh it after the action is executed.
                res = self._fallback_state_with_query(
                    sequence[-3:], reason="live_environment_state_deferred",
                )
                if step == 0 and self.logger:
                    self.logger.info(
                        "State prediction disabled for open-loop plan; "
                        "successor state will come from env.step"
                    )
            current_sars['next state'] = res['state']
            current_sars['done or termination'] = res['task_completion']
            done = False
            
            sequence[-1] = current_sars
            next_sars = {
                'state': res['state'],
                'action': '<PREDICT>',
                'reward (env response)': '<PREDICT>',
                'next state': None,
                'done or termination': None
            }
            
            plan_printout += f"\n<Action> {sequence[-1]['action']} - <Response>{sequence[-1]['reward (env response)']}"     
            sequence.append(next_sars)
            
            step += 1
            
        return sequence, plan_printout, self.token_sent, self.token_received

    @staticmethod
    def _sequence_context_text(sequence) -> str:
        for s in reversed(sequence or []):
            if not isinstance(s, dict):
                continue
            for key in ("next state", "state"):
                st = s.get(key)
                if isinstance(st, dict) and st.get("observation"):
                    inv = st.get("inventory", "")
                    return f"{st['observation']} {inv}".lower()
        return ""

    @staticmethod
    def _state_prediction_is_usable(result: dict) -> bool:
        """Return whether a model result contains a usable successor state."""
        if not isinstance(result, dict):
            return False
        state = result.get("state")
        if not isinstance(state, dict):
            return False
        observation = state.get("observation")
        if not isinstance(observation, str) or not observation.strip():
            return False
        return observation.strip().lower() not in {"pending", "unknown", "..."}

    @staticmethod
    def _fallback_state_with_query(sequence, reason: str = "") -> dict:
        """Carry forward the latest known state without inventing a transition.

        ``get_trajectory`` is an open-loop candidate planner and cannot call
        ``env.step``.  A failed prediction must therefore not fabricate a new
        observation; the execution loop will refresh it from the live env.
        """
        st = {}
        for s in reversed(sequence or []):
            if not isinstance(s, dict):
                continue
            # Prefer the current state over a speculative next state.  The
            # current SARSA item has next state='<PREDICT>'.
            for key in ("state", "next state"):
                candidate = s.get(key)
                if KnowledgeGraph._state_prediction_is_usable({"state": candidate}):
                    st = dict(candidate)
                    break
            if st:
                break
        result = {
            "query": None,
            "state": {
                "observation": st.get("observation", ""),
                "inventory": st.get("inventory", ""),
                "valid_receptacles": st.get("valid_receptacles", []),
            },
            "task_completion": False,
            "_state_prediction_fallback": True,
        }
        if reason:
            result["_state_prediction_fallback_reason"] = reason
        return result

    def _state_prediction_enabled(self) -> bool:
        """Whether open-loop planning should spend a call predicting state."""
        exp = getattr(self.agent, "agent_config", {}) or {}
        execution = exp.get("EXECUTION") or {}
        value = execution.get("cwme_state_prediction_enabled")
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip().lower() not in {"0", "false", "no", "off"}
        return bool(value)

    def _planner_replace_with_navigation(
        self,
        res: dict,
        rejected: str,
        task_str: str,
        ctx_text: str,
        recent_planned: set,
        reason: str,
    ) -> None:
        sub = planner_substitute_navigation(
            task_str, ctx_text, recent_planned, substance_search=True,
        )
        res["predicted_action"] = sub
        if not res.get("predicted_response"):
            res["predicted_response"] = f"Navigate or explore toward the task ({sub})."
        if self.logger:
            self.logger.info(
                f"Planner {reason}: replaced {rejected!r} with {sub!r}"
            )

    def plan_next_action_reward(self, env_sum, sequence, model=None, max_query=3):
        plan_model = model or self.cognition_model
        predicted = False
        num_query = 0
        kg_log = """"""
        ctx_text = self._sequence_context_text(sequence)
        task_str = env_sum.get("task", "") if isinstance(env_sum, dict) else ""
        plan_attempts = 0
        # The execution layer has an admissible-action fallback; do not spend
        # a long retry loop on a planner response that cannot be grounded.
        max_plan_attempts = 3
        while not predicted:
            plan_attempts += 1
            prev_planned = [
                s.get("action")
                for s in (sequence or [])
                if isinstance(s, dict) and s.get("action") not in (None, "<PREDICT>")
            ]
            recent_planned = {
                (a or "").strip().lower().replace("green house", "greenhouse")
                for a in prev_planned[-3:]
            }
            anti_repeat = ""
            if prev_planned:
                anti_repeat = (
                    f"\n\nAlready planned in this trajectory (do not repeat the last action): "
                    f"{prev_planned[-3:]}"
                )
            nav_hint = planner_phase_navigation_hint(task_str, ctx_text)
            if nav_hint:
                anti_repeat += f"\n\n{nav_hint}"
            planner_ctx = self._planner_context_block()
            planner_block = f"\n\n{planner_ctx}" if planner_ctx else ""
            action_gen_prompt = (
                self.action_gen.format(env_sum, sequence, kg_log)
                + planner_block
                + anti_repeat
                + "\n\nReturn ONLY one JSON object matching the required schema. "
                + 'Keys: "query", "predicted_action", "predicted_response", "reasoning". '
                + "All keys must use double quotes. No markdown fences, no preamble, no template text."
            )
            res, token_count = self._llm_retry(
                "plan_action",
                sm.ActionPrediction,
                action_gen_prompt,
                max_attempts=3,
                json=True,
                schema=sm.ActionPrediction,
            )
            self.token_sent += token_count['sent']
            self.token_received += token_count['received']
            if res['query']:
                num_query += 1
                if num_query >= max_query:
                    qa = f"""
QUERY: {res['query']}
RESPONSE: 'maximum number of query reached. Please pick an arbitrary action that is relevant to the task' 
                    """
                elif self._runtime_kg_query_enabled(): 
                    qa = f"""
QUERY: {res['query']}
RESPONSE: {self.query(res['query'])} 
                    """
                else:
                    qa = ""
                kg_log += qa
            if res.get("predicted_action"):
                act = str(res["predicted_action"]).strip()
                replace, replace_reason = planner_should_replace_action(
                    act, task_str, ctx_text,
                )
                if replace:
                    self._planner_replace_with_navigation(
                        res, act, task_str, ctx_text, recent_planned,
                        reason=replace_reason,
                    )
                    predicted = True
                else:
                    predicted = True
            elif not res.get("predicted_action") and plan_attempts >= 2:
                sub = planner_substitute_navigation(
                    task_str, ctx_text, recent_planned, substance_search=True,
                )
                res["predicted_action"] = sub
                res["predicted_response"] = (
                    res.get("predicted_response") or f"Execute {sub} toward the task."
                )
                if self.logger:
                    self.logger.info(
                        f"Planner empty action substitution: {sub!r}"
                    )
                predicted = True
            if not predicted and plan_attempts >= max_plan_attempts:
                sub = planner_substitute_navigation(
                    task_str, ctx_text, recent_planned, substance_search=True,
                )
                res["predicted_action"] = sub
                res["predicted_response"] = (
                    res.get("predicted_response") or f"Execute {sub} toward the task."
                )
                if self.logger:
                    self.logger.warning(
                        f"Planner fallback after {max_plan_attempts} attempts: {sub!r}"
                    )
                predicted = True
        return res
        
    def plan_next_state_termination(self, env_sum, sequence, model=None, max_query=3):
        plan_model = model or self.cognition_model
        kg_log = """"""
        predicted = False
        num_query = 0
        state_attempts = 0
        # A predicted state is only a hint. The live environment supplies the
        # truth after execution, so keep this bounded and degrade gracefully.
        max_state_attempts = 2
        last_query = ""
        res = self._fallback_state_with_query(sequence, reason="not_attempted")
        while not predicted:
            state_attempts += 1
            state_gen_prompt = (
                self.state_gen.format(sequence, kg_log)
                + '\n\nReturn one JSON object: {"query":null,"state":{"observation":"...",'
                + '"inventory":"","valid_receptacles":[]},"task_completion":false}. '
                + "JSON only, no other text."
            )
            task = str((env_sum or {}).get("task", ""))
            state_gen_prompt += (
                f"\nTask: {task}\n"
                "The environment has not been stepped by you. Do not invent a "
                "new room or object. If the latest KG answer is insufficient, "
                "return the latest known state with query=null."
            )
            if kg_log:
                state_gen_prompt += (
                    "\nA KG query was already answered above. Do not ask the same "
                    "query again; emit a concrete state object now."
                )
            if state_attempts > 1:
                state_gen_prompt += (
                    "\nPrevious state output was unusable. This is the final "
                    "attempt: query must be null and state.observation must be "
                    "a non-empty string."
                )
            try:
                res, token_count = self._llm_retry(
                    "plan_state",
                    sm.StateWithQuery,
                    state_gen_prompt,
                    max_attempts=2,
                    json=True,
                    schema=sm.StateWithQuery,
                )
                self.token_sent += token_count['sent']
                self.token_received += token_count['received']
            except ValueError as exc:
                if self.logger:
                    self.logger.warning(
                        "State prediction JSON failed (attempt %s): %s",
                        state_attempts,
                        exc,
                    )
                if state_attempts >= max_state_attempts:
                    res = self._fallback_state_with_query(
                        sequence, reason="invalid_json",
                    )
                    predicted = True
                continue
            query = str(res.get('query') or '').strip()
            if query:
                query_key = " ".join(query.lower().split())
                # A repeated query cannot add information. Let the next prompt
                # request a concrete state; if that still fails, carry forward
                # the grounded state below.
                if query_key == last_query or num_query >= max_query:
                    if self.logger:
                        self.logger.warning(
                            "State prediction query repeated/exhausted; "
                            "requesting state-only output",
                        )
                    query = ""
                elif not self._runtime_kg_query_enabled():
                    res = self._fallback_state_with_query(
                        sequence, reason="kg_query_disabled",
                    )
                    if self.logger:
                        self.logger.warning(
                            "State prediction fallback: KG query disabled"
                        )
                    predicted = True
                    break
                else:
                    last_query = query_key
                    num_query += 1
                    qa = f"""
QUERY: {query}
RESPONSE: {self.query(query) if self._runtime_kg_query_enabled() else ''}
                """
                    kg_log += qa
                    continue
            if self._state_prediction_is_usable(res) and not query:
                predicted = True
            elif state_attempts >= max_state_attempts:
                res = self._fallback_state_with_query(
                    sequence, reason="unusable_model_state",
                )
                if self.logger:
                    self.logger.warning(
                        "State prediction degraded after %s attempts; "
                        "carrying forward latest grounded state",
                        max_state_attempts,
                    )
                predicted = True
        return res
    
    
    def memory_reset(self):
        try:
            uuid = self.agent.this_uuid
            conn = self.connection_pool.getconn()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            cur.execute("SET session_replication_role = 'replica';")

            cur.execute("""
                DELETE FROM fact_tuples
                WHERE agent_uuid = %s;
            """, (uuid,))

            cur.execute("""
                DELETE FROM entities
                WHERE id IN (
                    SELECT e.id
                    FROM entities e
                    LEFT JOIN fact_tuples ft ON e.id = ft.source_entity_id OR e.id = ft.target_entity_id
                    WHERE ft.agent_uuid = %s
                    GROUP BY e.id
                    HAVING COUNT(ft.id) = 0
                );
            """, (uuid,))

            cur.execute("""
                DELETE FROM relationships
                WHERE id NOT IN (
                    SELECT DISTINCT relationship_id FROM fact_tuples
                );
            """)

            cur.execute("SET session_replication_role = 'origin';")
            conn.commit()

        except Exception as e:
            if conn:
                conn.rollback()
            print(f"Error removing data by UUID: {e}")
        finally:
            if cur:
                cur.close()
            if conn:
                self.connection_pool.putconn(conn)
