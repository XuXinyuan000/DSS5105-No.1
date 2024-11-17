from IPython.display import display, HTML, Markdown
from PIL import Image
import io
import base64
import numpy as np
import pandas as pd
import time
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Union
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import re
import gmft
from gmft.pdf_bindings import PyPDFium2Document
from gmft.auto import CroppedTable, TableDetector
from gmft.auto import AutoTableFormatter
from gmft.auto import AutoFormatConfig
import asyncio
from datetime import datetime
from openai import AsyncOpenAI 
import nest_asyncio
from pdfminer.high_level import extract_text
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from werkzeug.utils import secure_filename

def process_pdf(file_path):
    detector = TableDetector()

    config = AutoFormatConfig()
    config.semantic_spanning_cells = True
    config.enable_multi_header = True
    formatter = AutoTableFormatter(config)

    def ingest_pdf(pdf_path) -> list[CroppedTable]:
        doc = PyPDFium2Document(pdf_path)

        tables = []
        for page in doc:
            tables += detector.extract(page)
        return tables, doc

    _total_detect_time = 0
    _total_detect_num = 0
    _total_format_time = 0
    _total_format_num = 0

    #pdf_path = request.json['pdf_path']
    table_dict = {} 
    pdf_path = input("Enter the path to the PDF file: ")
    # detecting and extracting forms from the PDF
    start = time.time()
    tables, doc = ingest_pdf(pdf_path)
    num_pages = len(doc)
    end_detect = time.time()

    # Formatting each table and storing it in a dictionary
    for i, table in enumerate(tables):
        ft = formatter.extract(table)
        try:
            df = ft.df()  
            table_dict[f"table_{i+1}"] = df  
        except Exception as e:
            print(e)
            table_dict[f"table_{i+1}"] = None

    end_format = time.time()
    doc.close()

    # Output some time statistics
    print(f"Format time: {end_format - end_detect:.3f}s for {len(tables)} tables\n")
    _total_detect_time += end_detect - start
    _total_detect_num += num_pages
    _total_format_time += end_format - end_detect
    _total_format_num += len(tables)

    print(f"Macro: {_total_detect_time/_total_detect_num:.3f} s/page and {_total_format_time/_total_format_num:.3f} s/table.")
    print(f"Total: {(_total_detect_time+_total_format_num)/(_total_detect_num)} s/page")

    class DataFrameConverter:
        def __init__(self):
            pass

        def process_tables(self, tables_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
            if not tables_dict:
                raise ValueError("Input dict is empty")
                
            processed_tables = {}
            
            #Convert numeric columns
            for table_name, df in tables_dict.items():
                try:
                    processed_df = self._convert_numeric_columns(df)
                    processed_tables[table_name] = processed_df
                    
                except Exception as e:
                    print(f"An error occurred while processing table {table_name}: {str(e)}")
                    continue
                    
            return processed_tables

        def _convert_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            for col in df.columns:
                # Check if the column may contain values
                non_empty_values = df[col][df[col].notna()]
                if len(non_empty_values) == 0:
                    continue
                    
                try:
                    temp_series = non_empty_values.astype(str).apply(self._clean_numeric_string)
                    
                    if temp_series.apply(lambda x: self._is_numeric(x)).all():
                        df[col] = df[col].apply(
                            lambda x: self._clean_numeric_string(str(x)) if pd.notna(x) else x
                        )
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    continue
                    
            return df

        def _clean_numeric_string(self, value: str) -> str:
            if not isinstance(value, str):
                return value
                
            # Remove thousands separator
            value = value.replace(',', '')
            if '%' in value:
                try:
                    value = str(float(value.replace('%', '')) / 100)
                except:
                    pass
                    
            value = re.sub(r'\([^)]*\)', '', value)
            value = re.sub(r'[^0-9.-]', '', value)
            
            return value.strip()

        def _is_numeric(self, value: str) -> bool:
        
            if not value:
                return False           
            try:
                float(value)
                return True
            except ValueError:
                return False

        def check_conversion_results(self, tables_dict: Dict[str, pd.DataFrame]) -> None:
        
            for table_name, df in tables_dict.items():
                print(f"\nData type information for the table {table_name}:")
                print(df.dtypes)


    converter = DataFrameConverter()
    processed_tables = converter.process_tables(table_dict)
    converter.check_conversion_results(processed_tables)

    @dataclass
    class ESGIndicator:
        aspect: str
        kpi: str
        topic: str
        quantity: str
        search_terms: List[str]
        knowledge: str

    class ESGKnowledgeBase:
        def __init__(self):
            # Initialising the ESG Knowledge Base
            self.embedding_model = SentenceTransformer('moka-ai/m3e-base')
            self.vector_dim = 768
            
            # Initialising a vector index
            self.metadata_index = faiss.IndexFlatIP(self.vector_dim)
            self.table_index = faiss.IndexFlatIP(self.vector_dim)
            
            # memory structure
            self.indicators = []
            self.indicator_embeddings = [] 
            self.term_to_indicator = {}  
            self.tables = {}
            self.table_vectors = []
            self.table_metadata = []

        # Generate separate embedding vectors for each search term
        def _generate_indicator_embedding(self, indicator: ESGIndicator) -> List[np.ndarray]:

            embeddings = []
            
            
            for term in indicator.search_terms:
                # Make sure term is not an empty string
                if term.strip():  
                    term_embedding = self.embedding_model.encode([term.strip()])[0]
                    term_embedding = term_embedding.astype(np.float32)
                    term_embedding = term_embedding.reshape(1, -1)
                    faiss.normalize_L2(term_embedding)
                    embeddings.append(term_embedding)
            
            return embeddings

        def load_metadata(self):
            print("Loading metadata from Excel...")
            try:
                data = {
                    "aspect": [
                        "environment", "environment", "environment", "environment", "social",
                        "social", "social", "social", "social", "social", "government"
                    ],
                    "kpi": [
                        "Total greenhouse gas emissions", "Total energy consumption", "Total waste generated",
                        "Total water consumption", "Current employees by age groups (30 -50)",
                        "Current employees by female", "Current employees by male",
                        "Total number of employees", "turnover by female", "turnover by male",
                        "Women on the board"
                    ],
                    "topic": [
                        "total", "total", "total", "total", "age:30-50",
                        "female", "male", "total", "female", "male", "female"
                    ],
                    "quantity": [
                        "[absolute values]", "[absolute values]", "[absolute values]",
                        "[absolute values]", "[absolute values]", "[absolute values]",
                        "[absolute values]", "[absolute values]", "[absolute values]", 
                        "[absolute values]", "[absolute values]"
                    ],
                    "search_terms": [
                        "Greenhouse gases,Carbon dioxide,CO2,Carbon dioxide equivalents,GHG emission,Total CO2e (mt),total CO2e,carbon dioxide equivalents, Nitrogen oxides, Sox, sulfur dioxide, particulate matter, PM, inhalable suspended particles,Greenhouse Gas (GHG) Emissions (metric tons CO2e)2",
                        "Total Energy (GWh),total energy consumption,Total Energy",
                        "Total Waste (mt),Total Waste Generated",
                        "Total water consumption(m3),Total Water Consumption",
                        "PERCENTAGE BY AGE GROUP (% OF TERMINATIONS)",
                        "Job Categories,Global Workforce Data", 
                        "Job Categories,Global Workforce Data",
                        "Job Categories,Global Workforce Data", 
                        "Terminations", 
                        "Terminations",
                        "EMPLOYEE AND BOARD DIVERSITY"
                    ],
                    "knowledge": [
                        "Metric tons of carbon dioxide equivalent (tCO2e) of relevant GHG emissions. Report the Total, Scope 1 and Scope 2 GHG emissions and, if appropriate, Scope 3 GHG emissions. GHG emissions should be calculated in line with internationally recognised methodologies (e.g. GHG Protocol).",
                        "Total energy consumption, in megawatt hours or gigajoules (MWhs or GJ), within the organisation.",
                        "Total weight of waste generated, in metric tons (t), within organisation and where possible, to include relevant information of waste composition (e.g. hazardous vs non-hazardous, recycled vs non-recycled).",
                        "Total water consumption, in megalitres or cubic metres (ML or m³), across all operations.",
                        "Percentage of existing employees by age group: 30-50 years old",
                        "The total number of existing female employees",
                        "The total number of existing male employees",
                        "Total number of employees as at end of reporting period",
                        "the total number of female employee turnover during the period",
                        "the total number of male employee turnover during the period",
                        "Percentage of women on boards of directors,the number of female board directors as a percentage of all directors."
                    ]
                }

                df = pd.DataFrame(data)
        
                # Processing each row of data
                for idx, row in df.iterrows():
                    # Handling search_terms: comma-separated strings
                    search_terms = [term.strip() for term in str(row['search_terms']).split(',') if term.strip()]
                    
                    indicator = ESGIndicator(
                        aspect=row['aspect'],
                        kpi=row['kpi'],
                        topic=row['topic'],
                        quantity=row['quantity'],
                        search_terms=search_terms,
                        knowledge=row['knowledge']
                    )
                    self.indicators.append(indicator)
                    
                    # Generate separate embedding vectors for each search term
                    term_embeddings = self._generate_indicator_embedding(indicator)
                    
                    # Store the embedding vectors for each term and establish mapping relationships
                    for i, embedding in enumerate(term_embeddings):
                        self.indicator_embeddings.append(embedding)
                        self.term_to_indicator[len(self.indicator_embeddings) - 1] = {
                            'indicator_idx': len(self.indicators) - 1,
                            'term': indicator.search_terms[i]
                        }
                
                # Add all term's embedding vectors to the index
                if self.indicator_embeddings:
                    self.indicator_embeddings = np.vstack(self.indicator_embeddings)
                    self.metadata_index.add(self.indicator_embeddings)
                
                print(f"Successfully loaded {len(self.indicators)} indicators")
                
            except Exception as e:
                print(f"Error loading metadata: {str(e)}")
                raise

        # Loading table data from a dictionary        
        def load_tables_from_dict(self, tables_dict: Dict[str, pd.DataFrame]):
            
            print("\nLoading tables from dictionary...")
            print(f"Found {len(tables_dict)} tables")
            
            for table_id, df in tqdm(tables_dict.items(), desc="Processing tables"):
                try:
                    self._process_table(df, table_id, f"table_{table_id}")
                except Exception as e:
                    print(f"\nError processing table {table_id}: {str(e)}")
                    continue
        
        # Handling of individual forms
        def _process_table(self, df: pd.DataFrame, table_id: str, file_path: str):
            self.tables[table_id] = df
            
            # Extract text and generate vectors
            texts = self._extract_table_texts(df)
            if texts:
                vectors = self._generate_embeddings(texts)
                
                # Add to Index
                self.table_index.add(vectors)
                
                # Storing metadata
                start_idx = len(self.table_vectors)
                for i, text in enumerate(texts):
                    self.table_metadata.append({
                        'table_id': table_id,
                        'text': text,
                        'vector_id': start_idx + i,
                        'file_path': file_path,
                        'column': self._get_column_for_text(df, text)
                    })
                
                self.table_vectors.extend(vectors)
        
        # Generating text embeddings
        def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
            embeddings = self.embedding_model.encode(texts)
            embeddings = embeddings.astype(np.float32)
            faiss.normalize_L2(embeddings)
            return embeddings
        
        # Extracting text from tables
        def _extract_table_texts(self, df: pd.DataFrame) -> List[str]:
            texts = []        
            texts.extend(str(col) for col in df.columns)
            
            # Extract cell text
            for col in df.columns:
                for value in df[col].astype(str):
                    if (not value.replace('.', '').replace('-', '').isdigit() and 
                        not pd.isna(value) and 
                        value.strip() != ''):
                        texts.append(value)
            
            return list(set(texts))
        
        # Find the name of the column where the text is located
        def _get_column_for_text(self, df: pd.DataFrame, text: str) -> str:
            if text in df.columns:
                return text
            
            for col in df.columns:
                if text in df[col].astype(str).values:
                    return col
            
            return None
        
        # Search for tables related to a given indicator
        def search_and_extract(self, indicator_index: int, top_k: int = 5, similarity_threshold: float = 0.5) -> List[Dict]:        
            if not (0 <= indicator_index < len(self.indicators)):
                raise ValueError("Invalid indicator index")
                
            indicator = self.indicators[indicator_index]
            
            # Get the vector index of all search terms for this indicator
            term_indices = [
                i for i, mapping in self.term_to_indicator.items()
                if mapping['indicator_idx'] == indicator_index
            ]
            
            if not term_indices:
                return []
            
            # Search using these term vectors
            term_vectors = np.array([self.indicator_embeddings[i] for i in term_indices])
            results = []
            seen_tables = set()
            
            for term_idx, term_vector in zip(term_indices, term_vectors):
                term = self.term_to_indicator[term_idx]['term']
                
                scores, indices = self.table_index.search(
                    term_vector.reshape(1, -1), 
                    top_k
                )
                
                for score, idx in zip(scores[0], indices[0]):
                    if idx >= 0 and score >= similarity_threshold:
                        metadata = self.table_metadata[idx]
                        table_id = metadata['table_id']
                        
                        if table_id not in seen_tables:
                            seen_tables.add(table_id)
                            results.append({
                                'indicator': indicator,
                                'search_term': term,
                                'table_id': table_id,
                                'similarity': float(score),
                                'matched_text': metadata['text'],
                                'source_column': metadata['column'],
                                'file_path': metadata['file_path']
                            })
            
            results.sort(key=lambda x: x['similarity'], reverse=True)
            
            return results[:top_k]
        
        # Save knowledge base to file
        def save_knowledge_base(self, save_dir: str):
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            indicators_data = [
                {
                    'aspect': ind.aspect,
                    'kpi': ind.kpi,
                    'topic': ind.topic,
                    'quantity': ind.quantity,
                    'search_terms': ind.search_terms,
                    'knowledge': ind.knowledge
                }
                for ind in self.indicators
            ]
            
            with open(save_dir / 'indicators.json', 'w') as f:
                json.dump(indicators_data, f, indent=2)
                
            np.save(save_dir / 'indicator_embeddings.npy', self.indicator_embeddings)
            np.save(save_dir / 'table_vectors.npy', np.array(self.table_vectors))
            
            with open(save_dir / 'term_to_indicator.json', 'w') as f:
                json.dump(self.term_to_indicator, f, indent=2)
            
            with open(save_dir / 'table_metadata.json', 'w') as f:
                json.dump(self.table_metadata, f, indent=2)
                
            for table_id, df in self.tables.items():
                df.to_csv(save_dir / f'table_{table_id}.csv', index=False)
                
            print(f"Knowledge base saved to {save_dir}")
        
        # Groups search results by indicator and returns structured data
        def organize_results(self, all_results: List[Dict]) -> Dict:
            # Grouping results by indicator_idx
            grouped_results = {}
            for result in all_results:
                indicator = result['indicator']
                indicator_idx = self.indicators.index(indicator)
                
                if indicator_idx not in grouped_results:
                    grouped_results[indicator_idx] = {
                        'aspect': indicator.aspect,
                        'topic': indicator.topic,
                        'kpi': indicator.kpi,
                        'search_terms': indicator.search_terms,
                        'knowledge': indicator.knowledge,
                        'results': []
                    }
                
                table_id = result['table_id']
                table_df = self.tables[table_id]
                result_copy = result.copy()
                del result_copy['indicator']  
                
                # Adding Table Content
                result_copy['table_content'] = {
                    'columns': list(table_df.columns),
                    'data': table_df.values.tolist(),
                    'matched_column_content': table_df[result['source_column']].tolist() if result['source_column'] else None
                }
                
                grouped_results[indicator_idx]['results'].append(result_copy)
            
            return grouped_results

    def main():
        kb = ESGKnowledgeBase()
        print("Creating new knowledge base...")
        kb.load_metadata()
        kb.load_tables_from_dict(processed_tables)
        
        # Collect all search results
        all_results = []
        print("\nExtracting values for all indicators:")
        for i in range(len(kb.indicators)):
            results = kb.search_and_extract(i, similarity_threshold=0.5)
            if results:
                all_results.extend(results)
                indicator = kb.indicators[i]
                print(f"\nFound {len(results)} results for indicator: {indicator.kpi}")
        
        organized_results = kb.organize_results(all_results)
        print(f"\nTotal results found: {len(all_results)}")
        print(f"Number of indicators with matches: {len(organized_results)}")
        
        return organized_results

    if __name__ == "__main__":
        results = main()


    class ESGPromptGenerator:
        # Initialise the ESG Prompt generator with preset information
        def __init__(self):
            self.system_prompt = """You are an expert in the field of ESG (Environmental, Social, and Governance). 
                Your task is to analyze reference content to answer questions, providing your responses in a structured format. 
                Please ensure each section of your response starts with a clear heading on a new line. 
                Please follow these steps for your analysis:
                Begin by interpreting the meaning of the data disclosed in the table, summarizing it in brief terms. 
                Then, be aware that the provided reference content may not be related to the question. 
                Assess whether the reference content is relevant to the question. If it is, extract all the data related to the question and provide your answer. 
                Your response should include: 
                (1) Whether the reference content discloses data relevant to the question, indicated by a 'disclosure' field with a value of 0 or 1. 
                (2) If relevant data exists, provide the disclosed data in the 'data field.
                (3) The relevant data and analysis in a clearly structured format
                (4)Each section should be separated by newlines for clarity"""

            self.preset_instruction = """When analyzing ESG indicators, follow these principles:

                1. Disclosure Assessment:
                - Full: Complete data with clear metrics and context
                - Partial: Some data present but missing key elements
                - None: No relevant quantitative or qualitative data found

                2. Data Analysis:
                - Focus on numerical values and their temporal context
                - Pay attention to measurement units and consistency
                - Note any trends or significant changes

                3. Target Evaluation:
                - Identify both short-term and long-term targets
                - Document baseline years and target years
                - Note any progress towards targets

                4. Action Analysis:
                - Focus on concrete initiatives and strategies
                - Note implementation status where mentioned
                - Identify any monitoring or verification processes

                Please analyze the data using both the expert knowledge provided and the reference content."""

        def generate_prompt(self, indicator_results: Dict) -> Dict[str, str]:
            # Create user prompt
            prompt = f"{self.preset_instruction}\n\n"
            prompt += "Based on the above instructions and the following information, analyze the ESG indicator disclosure:\n\n"
            
            # 1. Indicator Information
            prompt += f"Indicator Information:\n"
            prompt += f"Aspect: {indicator_results['aspect']}\n"
            prompt += f"Topic: {indicator_results['topic']}\n"
            prompt += f"KPI: {indicator_results['kpi']}\n"
            prompt += f"Search Terms: {', '.join(indicator_results['search_terms'])}\n\n"
            
            # 2. Expert Knowledge
            if 'knowledge' in indicator_results:
                prompt += "Expert Knowledge:\n"
                prompt += f"{indicator_results['knowledge']}\n\n"
            
            # 3. Reference Content
            prompt += "Retrieved Table Content:\n"
            for i, result in enumerate(indicator_results['results'], 1):
                prompt += f"\nTable {i}:\n"
                prompt += f"Matched Text: {result['matched_text']}\n"
                prompt += f"Similarity Score: {result['similarity']}\n"
                columns = result['table_content']['columns']
                data = result['table_content']['data']
                prompt += f"Columns: {', '.join(columns)}\n"
                prompt += "Data Sample:\n"
                prompt += f"|{' | '.join(columns)}|\n"
                prompt += f"|{'|'.join(['---' for _ in columns])}|\n"

                for row in data: 
                    formatted_row = [str(cell) for cell in row]
                    prompt += f"|{' | '.join(formatted_row)}|\n"
                prompt += "\n"
            
            # 4. Question
            prompt += f"""
                The question is: Please answer based on the above information and do not strip away the given materials. 
                In terms of <{indicator_results['aspect']}>, extract the <{indicator_results['topic']}> about <{indicator_results['kpi']}> in 2023, and output <Quantity>.
                If no year information is displayed in the table, the default table is 2023 data
                """

            # 5. Answer Format Instruction
            prompt += """
                Please provide your analysis in the following structured format:

                <Disclosure>
                [Full/Partial/None] - Provide a detailed explanation of the disclosure level, referencing specific data points and any gaps.

                <KPI>
                KPI

                <Topic>
                topic

                <Value>
                - Year: [specify year]
                - Reported Value: [specific value]
                - Calculation Method: [if available]

                <Unit>
                - Primary unit of measurement
            
                For your analysis:
                1. Please ensure each section begins on a new line and is clearly separated from others.
                2. If any category lacks information in the provided data, explicitly state "No information available" for that category.
                3. Reported Value Outputs only numeric values, without any text, and converts them to float format.
                """
            
            return {
                "system_prompt": self.system_prompt,
                "user_prompt": prompt
            }

    class ESGLLMInterface:
        # Initialising the LLM interface
        def __init__(self, api_key: str, model: str = "gpt-4"):
            self.client = AsyncOpenAI(api_key=api_key)  
            self.model = model
            
        async def get_analysis(self, prompts: Dict[str, str]) -> str:
            max_retries = 3
            base_delay = 20 
            
            for attempt in range(max_retries):
                try:
                    response = await self.client.chat.completions.create(  
                        model=self.model,
                        messages=[
                            {"role": "system", "content": prompts["system_prompt"]},
                            {"role": "user", "content": prompts["user_prompt"]}
                        ],
                        temperature=0.3,
                        max_tokens=1500
                    )
                    # Format the response
                    content = response.choices[0].message.content
                    # Ensure consistent formatting of sections
                    formatted_content = self.format_response(content)
                    return formatted_content
                
                except Exception as e:
                    error_str = str(e)
                    if "rate_limit" in error_str.lower():
                        # Extracting suggested wait times from error messages
                        try:
                            wait_time = float(re.search(r'try again in (\d+\.?\d*)s', error_str).group(1))
                            wait_time = wait_time + 1  # Add an extra 1 second as a buffer
                        except:
                            wait_time = base_delay * (attempt + 1)  # If extraction is not possible, use an incremental wait time
                            
                        print(f"Rate limit reached. Waiting for {wait_time} seconds before retry...")
                        await asyncio.sleep(wait_time)
                        continue
                        
                    # Handling connection errors
                    elif "connection error" in error_str.lower():
                        wait_time = base_delay * (attempt + 1)
                        print(f"Connection error. Retrying in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                        continue
                        
                    else:
                        print(f"Error in LLM call: {error_str}")
                        if attempt < max_retries - 1:
                            wait_time = base_delay * (attempt + 1)
                            print(f"Retrying in {wait_time} seconds...")
                            await asyncio.sleep(wait_time)
                            continue
                        return None
                        
            print(f"Failed after {max_retries} attempts")
            return None

        # Format the response to ensure consistent spacing between sections
        def format_response(self, content: str) -> str:
            sections = ['<Disclosure>', '<KPI>', '<Topic>', '<Value>', '<Unit>']
            formatted_parts = []
            current_section = []
            
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line in sections:
                    if current_section:
                        formatted_parts.append('\n'.join(current_section))
                        formatted_parts.append('')  # Add empty line between sections
                    current_section = [line]
                elif line:
                    current_section.append(line)
                    
            if current_section:
                formatted_parts.append('\n'.join(current_section))
                
            return '\n\n'.join(formatted_parts)

    class ESGAnalyzer:
        def __init__(self, api_key: str):
            self.prompt_generator = ESGPromptGenerator()
            self.llm_interface = ESGLLMInterface(api_key)
            self.analysis_results = {}
        
        # Processing of individual indicator data
        async def process_single_indicator(self, indicator_idx: str, indicator_data: Dict) -> None:
            try:
                prompts = self.prompt_generator.generate_prompt(indicator_data)
                analysis = await self.llm_interface.get_analysis(prompts)
                
                if analysis:
                    self.analysis_results[indicator_idx] = {
                        "indicator_info": {
                            "aspect": indicator_data['aspect'],
                            "topic": indicator_data['topic'],
                            "kpi": indicator_data['kpi'],
                            "knowledge": indicator_data.get('knowledge', '')
                        },
                        "analysis": analysis
                    }
                    print(f"✓ Completed analysis for indicator {indicator_idx}")
                else:
                    print(f"✗ Failed to get analysis for indicator {indicator_idx}")
                    
            except Exception as e:
                print(f"✗ Error processing indicator {indicator_idx}: {str(e)}")
        
        # Prcessing of all indicators
        async def process_all_indicators(self, grouped_results: Dict) -> Dict:
            semaphore = asyncio.Semaphore(3)
            
            async def process_with_semaphore(idx, data):
                async with semaphore:
                    await self.process_single_indicator(idx, data)
                    await asyncio.sleep(1)
            
            tasks = []
            for indicator_idx, indicator_data in grouped_results.items():
                tasks.append(process_with_semaphore(indicator_idx, indicator_data))
            
            await asyncio.gather(*tasks, return_exceptions=True)
            return self.analysis_results

    async def main():
        analyzer = ESGAnalyzer("api_key")
        analysis_results = await analyzer.process_all_indicators(results)

        return analysis_results

    # 运行
    #results_values1 = await main()
    if __name__ == "__main__":
        results_values1 = asyncio.run(main())
        print(results_values1)


    def detect_column_layout(page):
        text = page.extract_text()
        if text is None:
            return 1
        
        # Get the left margin position of the text block
        left_margins = [char['x0'] for char in page.chars if 'x0' in char]
        if not left_margins:
            return 1
        
        # Clustering detects the left margin position and infers the number of columns
        left_margins = sorted(left_margins)
        threshold = 50
        columns = [left_margins[0]]
        
        for margin in left_margins[1:]:
            if margin - columns[-1] > threshold:
                columns.append(margin)
        
        # The number of columns is the number of detected columns
        return len(columns)

    def extract_pdf_text(pdf_path):
        text_per_page = {}
        
        # Extract text using pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            for pagenum, page in enumerate(pdf.pages):
                page_text = []
                page_images_text = []
                
                # Detect the number of layout
                column_count = detect_column_layout(page)
                print(f"\rProcessing: {pagenum + 1}/{total_pages}", end="")
                
                # Split page based on number of columns
                width = page.width
                if column_count > 1:
                    column_width = width / column_count
                    for i in range(column_count):
                        left = max(i * column_width - 10, 0)  
                        right = min((i + 1) * column_width + 10, width)
                        bbox = (left, 0, right, page.height)
                        column_text = page.within_bbox(bbox).extract_text() or ""
                        page_text.append(column_text)
                else:
                    # Single column layout: extract text directly
                    page_text.append(page.extract_text())
                
                # If there are pictures or graphic elements, perform OCR processing
                images = convert_from_path(pdf_path, first_page=pagenum+1, last_page=pagenum+1)
                for image in images:
                    image_text = pytesseract.image_to_string(image)
                    page_images_text.append(image_text)
                
                # Store extracted content for each page
                text_per_page[f"Page_{pagenum + 1}"] = {
                    'images_text': page_images_text
                }
        print("\r" + " " * 50 + "\r", end="")
        return text_per_page
        
    text = extract_pdf_text(pdf_path)


    # Count the number of occurrences of keywords in each page of result and return as a DataFrame
    def count_keywords_in_result(result, keywords):
        data = []
        for page, content in result.items():
            # Splice OCR text on each page
            combined_text = " ".join(content['images_text'])
            
            # Counting keyword occurrences
            keyword_counts = {keyword: len(re.findall(keyword, combined_text)) for keyword in keywords}
            total = sum(keyword_counts.values())

            # Record results only if the total is greater than 0
            if total > 0:
                data.append({
                    "KPI": "Anti-corruption training for employees",
                    "Year": "2023",
                    "Reported Value": total,
                    "Unit": "Not found"
                })

        return pd.DataFrame(data)

    keywords = ["205-1", "205-2", "205-3"]
    df_keywords = count_keywords_in_result(text, keywords)

    def search_assurance_info(result):
        """
        In the extracted PDF text content, retrieve external independent guarantee, internal guarantee, and non-guaranteed information,
        and determine the scope of the guarantee. Outputs the result in DataFrame format.
        """
        # Define keywords
        external_keywords = ["external independent assurance", "third-party assurance", "independent audit", "independent", "external"]
        internal_keywords = ["internal assurance", "internal audit", "internally reviewed", "assured by internal team", "internal"]
        no_assurance_keywords = ["no assurance", "without assurance", "no external verification"]
        scope_keywords = ["Project Type", "Number of Projects", "Allocated"]

        # Initialize result variables
        assurance_type = "No"
        assurance_scope = set()

        # Iterate over each page
        for content in result.values():
            # Combine the text from each page
            text = "\n".join(content['images_text'])
            
            # Check for assurance types
            if assurance_type == "No":
                if any(re.search(keyword, text, re.IGNORECASE) for keyword in external_keywords):
                    assurance_type = 1
                elif any(re.search(keyword, text, re.IGNORECASE) for keyword in internal_keywords):
                    assurance_type = 0
                elif any(re.search(keyword, text, re.IGNORECASE) for keyword in no_assurance_keywords):
                    assurance_type = "no assurance"

            # Check and collect assurance scope
            for keyword in scope_keywords:
                if re.search(keyword, text, re.IGNORECASE):
                    assurance_scope.add(keyword)

        # Determine the "Yes or No" field
        has_assurance = "Yes" if assurance_type != "No" else "No"
        
        # Prepare the output data in the required format
        data = {
            "KPI": "Assurance of sustainability report",
            "Year": "2023",
            "Reported Value": assurance_type,
            "Unit": "Not found"
        }
        
        # Convert data to a DataFrame
        return pd.DataFrame([data])

    # Generate and display the DataFrame
    df_assurance = search_assurance_info(text)

    combined_df1 = pd.concat([df_keywords, df_assurance], ignore_index=True)
    print(combined_df1)

    def process_results_dict(results_dict):
        # List to store processed results
        processed_results = []
        
        for key, value in results_dict.items():
            # Get indicator info
            indicator_info = value.get('indicator_info', {})
            kpi = indicator_info.get('kpi', 'Not found')
            knowledge = indicator_info.get('knowledge', '')
            
            # Get analysis text
            analysis_text = value.get('analysis', '')
            
            # Extract year
            year_match = re.search(r"Year:\s*(\d{4})", analysis_text)
            year = year_match.group(1) if year_match else "Not found"
            
            # Extract reported value
            reported_value_match = re.search(r"Reported Value:\s*([0-9]*\.?[0-9]+(?:\s*\(or\s*[0-9]*\.?[0-9]+%\))?)", analysis_text)
            reported_value = reported_value_match.group(1) if reported_value_match else "Not found"
            
            # Extract unit from knowledge field
            unit_match = re.search(r"\(([^)]+)\)", knowledge)
            unit = unit_match.group(1) if unit_match else "Not found"
            
            # Store results
            processed_results.append({
                "KPI": kpi,
                "Year": year,
                "Reported Value": reported_value,
                "Unit": unit
            })
        
        # Convert to DataFrame
        return pd.DataFrame(processed_results)

    combined_df2 = process_results_dict(results_values1)
    print(combined_df2)

    combined_df = pd.concat([combined_df2, combined_df1], ignore_index=True)
    print(combined_df)

    combined_df['Reported Value'] = pd.to_numeric(combined_df['Reported Value'], errors='coerce')
    new_row1 = {
        'KPI': 'Current Employees by Gender (Female)',
        'Year': combined_df[combined_df['KPI'] == 'Current employees by female']['Year'].values[0], 
        'Reported Value': (
            combined_df[combined_df['KPI'] == 'Current employees by female']['Reported Value'].values[0] / 
            combined_df[combined_df['KPI'] == 'Total number of employees']['Reported Value'].values[0]
        ).round(2),  
        'Unit': 'Not found' 
    }
    new_row2 = {
        'KPI': 'Turnover by Gender (Female)',
        'Year': combined_df[combined_df['KPI'] == 'turnover by female']['Year'].values[0], 
        'Reported Value': (
            combined_df[combined_df['KPI'] == 'turnover by female']['Reported Value'].values[0] / (
            combined_df[combined_df['KPI'] == 'turnover by female']['Reported Value'].values[0] + combined_df[combined_df['KPI'] == 'turnover by male']['Reported Value'].values[0])
        ).round(2), 
        'Unit': 'Not found'  
    }

    combined_df3 = pd.concat([combined_df, pd.DataFrame([new_row1])], ignore_index=True)
    combined_df4 = pd.concat([combined_df3, pd.DataFrame([new_row2])], ignore_index=True)
    combined_df4 = combined_df4[~combined_df4['KPI'].isin(['turnover by female', 'turnover by male', 'Current employees by female','Current employees by male'])]

    dataframe = combined_df4[['KPI', 'Reported Value']]
    company = pdf_path.split("\\")[-1].split("_")[0] 
    year = pdf_path.split("_")[-1].split(".")[0]

    dataframe["Company Name"] = company
    dataframe["Year"] = year
    dataframe["Company Name"] = company
    dataframe["Year"] = year
    # Pivot the DataFrame to have KPIs as columns and the reported values as the data
    df_pivot = dataframe.pivot(index=["Company Name", "Year"], columns="KPI", values="Reported Value").reset_index()

    df_pivot.columns.name = None
    # Select specific columns for the final output
    df_pivot = df_pivot.rename(columns={"Total greenhouse gas emissions": "Absolute emissions scope (total)"})
    print(df_pivot)
    # Select specific columns for the final output
    esg_data = df_pivot[["Company Name", "Year", 'Absolute emissions scope (total)', 'Total water consumption', 
                'Turnover by Gender (Female)', 'Current Employees by Gender (Female)', 
                'Women on the board', 'Anti-corruption training for employees',"Total number of employees"]]
    #df = df_pivot.iloc[:, [0,1,7,10,11,5,13,2,9]]
    fill_values = {
        'Absolute emissions scope (total)': 72594.0,
        'Total water consumption': 37612.0,
        'Turnover by Gender (Female)': 0.56,
        'Current Employees by Gender (Female)': 0.53,
        'Women on the board': 0.36,
        'Anti-corruption training for employees': 3.0,
        'Total number of employees': 43145.0
    }

    # Replace missing values with the specified values
    esg_data = esg_data.fillna(value=fill_values)

    esg_data['Absolute emissions per person'] = esg_data['Absolute emissions scope (total)'] / esg_data['Total number of employees']
    esg_data['Water consumption per person'] = esg_data['Total water consumption'] / esg_data['Total number of employees']

    # Select the desired columns
    final_esg_data = esg_data[['Company Name', 'Year', 'Absolute emissions per person','Water consumption per person','Turnover by Gender (Female)', 
                'Current Employees by Gender (Female)', 
                'Women on the board', 'Anti-corruption training for employees']]
    final_esg_data

    def score_women_indep(value):
        if value == 1:
            return 10
        elif 0.8 <= value < 1:
            return 8
        elif 0.6 <= value < 0.8:
            return 6
        elif 0.4 <= value < 0.6:
            return 4
        elif 0.2 <= value < 0.4:
            return 2
        else:
            return 0
    def score_absolute_emissions(value):
        if value < 2:
            return 10
        elif 2 <= value < 5:
            return 8
        elif 5 <= value < 10:
            return 6
        elif 10 <= value < 15:
            return 4
        elif 15 <= value < 20:
            return 2
        else:
            return 0

    def score_water_consumption(value):
        # Add the specific scoring criteria for water consumption per person here
        # Assuming similar criteria as absolute emissions for now
        if value < 20:
            return 10
        elif 20 <= value < 40:
            return 8
        elif 40 <= value < 60:
            return 6
        elif 60 <= value < 80:
            return 4
        elif 80 <= value < 200:
            return 2
        else:
            return 0

    def Turnover(value):
        if value == 0:
            return 10
        elif 1 <= value < 2:
            return 8
        elif 2 <= value < 3:
            return 6
        elif 3 <= value < 5:
            return 4
        elif 5 <= value < 10:
            return 2
        else:
            return 0
    def Anti_corruption(value):
        if value >= 3:
            return 10
        elif value == 2:
            return 7
        elif value == 1:
            return 3
        else:
            return 0
    # Apply scoring to the columns
    score = final_esg_data.loc[:, ['Company Name', 'Year']]
    score_final = final_esg_data.loc[:, ['Company Name', 'Year']]
    score['Women on the board score'] = final_esg_data['Women on the board'].apply(score_women_indep)
    score['Absolute emissions per person score'] = final_esg_data['Absolute emissions per person'].apply(score_absolute_emissions)
    score['Water consumption per person score'] = final_esg_data['Water consumption per person'].apply(score_water_consumption)
    score['Turnover by Gender (Female)'] = final_esg_data['Turnover by Gender (Female)'].apply(Turnover)
    score['Current Employees by Gender (Female)'] = final_esg_data['Current Employees by Gender (Female)'].apply(score_women_indep)
    score['Anti-corruption training for employees'] = final_esg_data['Anti-corruption training for employees'].apply(Anti_corruption)
    score[['Company Name', 'Year', 'Absolute emissions per person score','Water consumption per person score','Turnover by Gender (Female)',
            'Current Employees by Gender (Female)',
            'Women on the board score','Anti-corruption training for employees']]
    score_final['Score E'] = 0.5 * score['Absolute emissions per person score'] + 0.5 * score['Water consumption per person score']
    score_final['Score S'] = 0.5* score['Turnover by Gender (Female)'] + 0.5* score['Current Employees by Gender (Female)']
    score_final['Score G'] = 0.5 * score['Anti-corruption training for employees'] + 0.5 * score['Women on the board score']
    score_final['Total ESG Score'] = 0.4 * score_final['Score E'] + 0.3 * score_final['Score S'] + 0.3 * score_final['Score G']

    # Define the rating function
    def assign_rating(value):
        if 8.571 <= value <= 10:
            return 'AAA'
        elif 7.143 <= value < 8.571:
            return 'AA'
        elif 5.714 <= value < 7.143:
            return 'A'
        elif 4.286 <= value < 5.714:
            return 'BBB'
        elif 2.857 <= value < 4.286:
            return 'BB'
        elif 1.429 <= value < 2.857:
            return 'B'
        else:
            return 'CCC'

    # Apply the rating to 'Total ESG Score' column
    score_final['Rating'] = score_final['Total ESG Score'].apply(assign_rating)
    #score = score_final[['Company Name','Year', 'Score E','Score S', 'Score G','Total ESG Score', 'Rating']]
    print(score_final)

    return df_pivot, score_final