#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import manticoresearch
from manticoresearch.rest import ApiException
from utils.jsonl_utils import jsonl_iter
from utils.kb_utils import enrich_results_with_kb
import json
import time


class ManticoreComparator:
    """
    Manticore Search-based vector search comparator for benchmarking.
    
    Supports cosine similarity search via HNSW index.
    """
    
    def __init__(self, dimension: int = 1024, host: str = "http://localhost:9308", 
                 cluster_name: str = "FTS_1", hnsw_m: int = 16, 
                 ef_construction: int = 200, ef_search: int = 2000,
                 cluster_nodes: Optional[List[str]] = None):
        """
        Initialize Manticore comparator.
        
        Args:
            dimension: Vector dimension (default: 1024)
            host: Manticore Search host URL (default: http://localhost:9308)
            cluster_name: Manticore cluster name (default: FTS_1)
            hnsw_m: HNSW M parameter - number of bi-directional links per node (default: 16)
            ef_construction: HNSW ef_construction parameter (default: 200)
            ef_search: HNSW ef_search parameter (default: 2000)
            cluster_nodes: List of cluster node URLs (default: localhost:9308, 9318, 9328)
        """
        self.dimension = dimension
        self.host = host
        self.cluster_name = cluster_name
        self.hnsw_m = hnsw_m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.index_name = None
        self.configuration = None
        self.query_log_path = None
        self.select_log_path = None
        self.ddl_log_path = None
        self.cluster_nodes = cluster_nodes or [
            "http://localhost:9308",
            "http://localhost:9318",
            "http://localhost:9328"
        ]
        
    def get_config(self, host: Optional[str] = None):
        """Get Manticore configuration."""
        return manticoresearch.Configuration(host=host or self.host)
    
    def index_exists(self, index_name: str) -> bool:
        """
        Check if an index exists.
        
        Args:
            index_name: Name of the index to check
        
        Returns:
            True if index exists, False otherwise
        """
        try:
            with manticoresearch.ApiClient(self.get_config()) as api_client:
                utils_api = manticoresearch.UtilsApi(api_client)
                result = utils_api.sql(f"SHOW TABLES LIKE '{index_name}'", raw_response=True)
                rows = result.actual_instance[0].get('data', [])
                return len(rows) > 0
        except Exception:
            return False
    
    def index_in_cluster(self, index_name: str) -> bool:
        """
        Check if an index is already part of the cluster.
        
        Args:
            index_name: Name of the index to check
        
        Returns:
            True if index is in cluster, False otherwise
        """
        try:
            with manticoresearch.ApiClient(self.get_config()) as api_client:
                utils_api = manticoresearch.UtilsApi(api_client)
                result = utils_api.sql(f"SHOW CLUSTER {self.cluster_name}", raw_response=True)
                rows = result.actual_instance[0].get('data', [])
                for row in rows:
                    if row.get('Index') == index_name or row.get('index') == index_name:
                        return True
                return False
        except Exception:
            return False
    
    def setup_schema(self, index_name: str, filter_type: Optional[str] = None):
        """
        Setup Manticore index schema.
        
        Args:
            index_name: Name of the index to create
            filter_type: Optional type filter for index name
        """
        self.index_name = index_name
        self.configuration = self.get_config()
        
        # Check if index already exists and is in cluster
        index_exists = self.index_exists(index_name)
        in_cluster = self.index_in_cluster(index_name) if index_exists else False
        
        if index_exists:
            print(f"Index {index_name} already exists")
            if in_cluster:
                print(f"Index {index_name} is already part of cluster {self.cluster_name}")
                return
        else:
            # Build schema
            schema = f"""
                type string,
                `timestamp` bigint,
                vector float_vector knn_type='hnsw' knn_dims='{self.dimension}' 
                hnsw_similarity='COSINE' hnsw_m='{self.hnsw_m}' hnsw_ef_construction='{self.ef_construction}'
            """
            
            index_settings = "engine='columnar' optimize_cutoff='100' rt_mem_limit='10240M'"
            
            try:
                with manticoresearch.ApiClient(self.configuration) as api_client:
                    utils_api = manticoresearch.UtilsApi(api_client)
                    create_index_statement = f"CREATE TABLE IF NOT EXISTS {index_name}({schema}) {index_settings}"
                    if self.ddl_log_path:
                        with open(self.ddl_log_path, "a") as log_file:
                            log_file.write(create_index_statement + ";\n")
                    utils_api.sql(create_index_statement, raw_response=True)
                    print(f"Created table {index_name}")
            except ApiException as e:
                print(f"Exception when creating table: {e}")
                raise e
            except Exception as e:
                print(f"Error creating table: {e}")
                raise
        
        print(f"DONE: Manticore -> setup schema for {index_name}")
    
    def remove_index_from_cluster(self, index_name: Optional[str] = None):
        """
        Remove an index from the cluster.
        First removes from cluster, then drops table on each cluster node.
        
        Args:
            index_name: Name of the index to remove. If None, uses self.index_name
        """
        table_name = index_name or self.index_name
        if not table_name:
            raise ValueError("No index name provided and no index_name set")
        
        # Step 1: Remove table from cluster
        print(f"Removing table {table_name} from cluster {self.cluster_name}...")
        try:
            with manticoresearch.ApiClient(self.get_config()) as api_client:
                utils_api = manticoresearch.UtilsApi(api_client)
                alter_cluster_query = f"ALTER CLUSTER {self.cluster_name} DROP {table_name}"
                if self.ddl_log_path:
                    with open(self.ddl_log_path, "a") as log_file:
                        log_file.write(alter_cluster_query + ";\n")
                utils_api.sql(alter_cluster_query, raw_response=True)
                print(f"Successfully removed {table_name} from cluster {self.cluster_name}")
        except ApiException as e:
            print(f"Warning: Could not remove table from cluster (may not be in cluster): {e}")
        except Exception as e:
            print(f"Warning: Error removing table from cluster: {e}")
        
        # Step 2: Drop table on each cluster node
        print(f"Dropping table {table_name} on each cluster node...")
        for node_host in self.cluster_nodes:
            try:
                with manticoresearch.ApiClient(self.get_config(node_host)) as api_client:
                    utils_api = manticoresearch.UtilsApi(api_client)
                    drop_query = f"DROP TABLE IF EXISTS {table_name}"
                    if self.ddl_log_path:
                        with open(self.ddl_log_path, "a") as log_file:
                            log_file.write(drop_query + ";\n")
                    utils_api.sql(drop_query, raw_response=True)
                    print(f"Successfully dropped {table_name} on {node_host}")
            except ApiException as e:
                print(f"Warning: Could not drop table on {node_host}: {e}")
            except Exception as e:
                print(f"Warning: Error dropping table on {node_host}: {e}")
        
        print(f"Completed removal of {table_name} from cluster")
    
    def _write_to_manticore(self, chunk: List[Dict], num_retries: int = 3, 
                           sleep_between_retries: int = 60):
        """Write a chunk of documents to Manticore."""
        configuration = self.get_config()

        def _sql_escape(value):
            if value is None:
                return "NULL"
            if isinstance(value, (int, float)):
                return str(value)
            text = str(value).replace("\\", "\\\\").replace("'", "\\'")
            return "'" + text + "'"

        def _format_vector(vector):
            return "(" + ",".join(str(v) for v in vector) + ")"

        values = []
        for document in chunk:
            values.append(
                "({id}, {type}, {timestamp}, {vector})".format(
                    id=_sql_escape(document.get("id")),
                    type=_sql_escape(document.get("type", "")),
                    timestamp=_sql_escape(document.get("timestamp")),
                    vector=_format_vector(document.get("vector", [])),
                )
            )

        insert_sql = "REPLACE INTO {index} (id, type, `timestamp`, vector) VALUES {values}".format(
            index=self.index_name,
            values=",".join(values),
        )

        if self.query_log_path:
            with open(self.query_log_path, "a") as log_file:
                log_file.write(insert_sql + ";\n")
        
        while num_retries:
            try:
                with manticoresearch.ApiClient(configuration) as api_client:
                    utils_api = manticoresearch.UtilsApi(api_client)
                    try:
                        utils_api.sql(insert_sql, raw_response=True)
                    except ApiException as e:
                        print(f"Exception when calling UtilsApi->sql: {e}")
                        raise e
                break
            except Exception as e:
                num_retries -= 1
                if num_retries > 0:
                    print(f"Write to manticore failed: {repr(e)}. {num_retries} retries left. "
                          f"Sleeping for {sleep_between_retries}s...")
                    time.sleep(sleep_between_retries)
                else:
                    raise
    
    def load_data(self, data_path: str, filter_type: Optional[str] = None, 
                  max_rows: Optional[int] = None, index_name: Optional[str] = None,
                  rebuild_index: bool = False, read_batch_size: int = 10):
        """
        Load data from JSONL file and index into Manticore.
        
        Args:
            data_path: Path to the JSONL data file
            filter_type: Optional type filter (e.g., 'x', 'y') to match queries
            max_rows: Optional maximum number of valid vectors to load (None = load all)
            index_name: Optional index name. If None, uses default based on data_path
            rebuild_index: If True, drop and recreate the index (default: False)
            read_batch_size: Batch size for reading and writing data (default: 10000)
        """
        # Determine index name
        if index_name is None:
            data_file = Path(data_path).stem.replace('.json', '').replace('.gz', '')
            filter_suffix = f"_{filter_type}" if filter_type else ""
            max_rows_suffix = f"_max{max_rows}" if max_rows else ""
            index_name = f"manticore_index_{data_file}{filter_suffix}{max_rows_suffix}"
        
        self.index_name = index_name
        
        # Check if index exists and handle rebuild
        if rebuild_index:
            print("Rebuild requested, removing existing index from cluster if it exists...")
            self.remove_index_from_cluster(index_name)
        
        # Setup schema
        self.setup_schema(index_name, filter_type)
        
        # Check if index already exists and has data (skip loading if it does, unless rebuild)
        if not rebuild_index and self.index_exists(index_name):
            # Check if index has any data
            try:
                with manticoresearch.ApiClient(self.get_config()) as api_client:
                    utils_api = manticoresearch.UtilsApi(api_client)
                    result = utils_api.sql(f"SELECT COUNT(*) as total FROM {index_name}", raw_response=True)
                    rows = result.actual_instance[0].get('data', [])
                    if rows and rows[0].get('total', 0) > 0:
                        count = rows[0].get('total', 0)
                        print(f"Index {index_name} already exists with {count} documents. Skipping data loading.")
                        print(f"Use --rebuild flag to reload data.")
                        return
            except Exception as e:
                print(f"Warning: Could not check index data count: {e}. Proceeding with data load...")
        
        print(f"Loading data from {data_path}...")
        if max_rows is not None:
            print(f"Limiting to {max_rows} vectors")
        
        start_time = time.time()
        
        # Index in batches sequentially (no multi-threading)
        batch = []
        valid_count = 0
        
        for item in tqdm(jsonl_iter(data_path), desc="Indexing vectors"):
            if max_rows is not None and valid_count >= max_rows:
                break
            
            if 'vector' not in item or not item['vector'] or len(item['vector']) == 0:
                continue
            
            if filter_type is not None and item.get('type') != filter_type:
                continue
            
            if len(item['vector']) != self.dimension:
                print(f"Warning: Skipping item {item.get('id')} - vector dimension mismatch "
                      f"(expected {self.dimension}, got {len(item['vector'])})")
                continue
            
            # Prepare document
            doc = {
                'id': item['id'],
                'type': item.get('type', ''),
                'vector': item['vector'],
                'timestamp': int(time.time())
            }
            
            batch.append(doc)
            valid_count += 1
            
            # Write batch when it reaches the batch size
            if len(batch) >= read_batch_size:
                self._write_to_manticore(batch)
                batch = []
        
        # Write remaining batch
        if batch:
            self._write_to_manticore(batch)
        
        print(f"Indexing done in {time.time() - start_time:.2f}s!")
        print(f"Indexed {valid_count} vectors to {index_name}")
    
    def search(self, query_vector: List[float], k: int = 50, 
               filter_type: Optional[str] = None,
               kb_base_url: Optional[str] = None) -> List[Dict]:
        """
        Search for k nearest neighbors.
        
        API Endpoint: {host}/sql (e.g., http://localhost:9308/sql)
        Uses Manticore Search UtilsApi.sql() method to execute SQL query with KNN search.
        
        Args:
            query_vector: Query vector (list of floats)
            k: Number of results to return (default: 50)
            filter_type: Optional type filter (e.g., 'x', 'y')
            kb_base_url: Optional KB API base URL to enrich results. If provided, results will include 'kb_data'
        
        Returns:
            List of dictionaries with 'id' and 'score' keys (and 'kb_data' if kb_base_url provided), sorted by score ascending
        """
        if self.index_name is None:
            raise ValueError("Index not set up! Call load_data() or setup_schema() first.")
        
        if len(query_vector) != self.dimension:
            raise ValueError(f"Query vector dimension mismatch (expected {self.dimension}, got {len(query_vector)})")
        
        # Build query vector string
        vector_str = ','.join(str(v) for v in query_vector)
        
        # Build WHERE clause - always need WHERE for KNN
        if filter_type:
            where_clause = f"WHERE type IN ('{filter_type}') AND knn (vector, {k}, ({vector_str}), {{ ef={self.ef_search} }})"
        else:
            where_clause = f"WHERE knn (vector, {k}, ({vector_str}), {{ ef={self.ef_search} }})"
        
        # Build KNN query
        query = f"""
            SELECT id, knn_dist() as score
            FROM {self.index_name}
            {where_clause}
            ORDER BY score ASC
            LIMIT {k}
        """
        if self.select_log_path:
            with open(self.select_log_path, "a") as log_file:
                log_file.write(query.strip() + ";\n")
        print("Manticore KNN SQL:\n{query}".format(query=query.strip()))
        
        # Run query on each cluster node and merge results
        all_results = {}  # Use dict to deduplicate by ID (keep best score)
        node_results = {}  # Store results per node
        
        for node_host in self.cluster_nodes:
            endpoint_url = f"{node_host}/sql"
            print(f"Querying Manticore Search node: {endpoint_url}")
            
            try:
                with manticoresearch.ApiClient(manticoresearch.Configuration(host=node_host)) as api_client:
                    utils_api = manticoresearch.UtilsApi(api_client)
                    result = utils_api.sql(query, raw_response=True)

                    rows = result.actual_instance[0].get('data', [])
                    
                    node_result_list = []
                    for row in rows:
                        doc_id = row.get('id')
                        score = float(row.get('score', 0.0))
                        
                        result_item = {
                            'id': doc_id,
                            'score': score
                        }
                        node_result_list.append(result_item)
                        
                        # Keep the best (lowest) score for each document ID
                        if doc_id not in all_results or score < all_results[doc_id]['score']:
                            all_results[doc_id] = result_item
                    
                    # Enrich results with KB data if kb_base_url is provided
                    node_result_list = enrich_results_with_kb(node_result_list, kb_base_url)
                    
                    node_results[node_host] = node_result_list
                    print(f"  Node {node_host}: Found {len(rows)} results")
                    for result in node_result_list[:10]:
                        print(result)
                    
            except ApiException as e:
                print(f"  Error querying node {node_host}: {e}")
                node_results[node_host] = []
                continue
            except Exception as e:
                print(f"  Unexpected error on node {node_host}: {e}")
                node_results[node_host] = []
                continue
        
        # Print summary of results per node
        print(f"\nResults summary per node:")
        for node_host, results_list in node_results.items():
            print(f"  {node_host}: {len(results_list)} results")
        
        # Convert dict to list and sort by score
        results = list(all_results.values())
        results.sort(key=lambda x: (x['score'], -x['id']))  # Sort by score ascending, then id descending
        
        # Enrich merged results with KB data if kb_base_url is provided
        if kb_base_url:
            results = enrich_results_with_kb(results[:k], kb_base_url)
            return results
        
        # Return top k results
        return results[:k]
    
    def search_with_kb(self, query_vector: List[float], k: int = 50,
                       filter_type: Optional[str] = None,
                       kb_base_url: Optional[str] = None) -> List[Dict]:
        """
        Search for k nearest neighbors and enrich results with KB entity information.
        
        Args:
            query_vector: Query vector (list of floats)
            k: Number of results to return (default: 50)
            filter_type: Optional type filter (e.g., 'x', 'y')
            kb_base_url: Base URL for KB API (optional)
        
        Returns:
            List of dictionaries with 'id', 'score', and 'kb_data' keys
        """
        # Perform vector search with KB enrichment
        return self.search(query_vector, k, filter_type, kb_base_url=kb_base_url)


def main():
    """Example usage."""
    parser = argparse.ArgumentParser(description='Manticore Comparator for KNN Benchmarking')
    parser.add_argument('--rebuild', action='store_true',
                       help='Force rebuild the index (drop and recreate)')
    parser.add_argument('--max-rows', type=int, default=50000,
                       help='Maximum number of rows to load (default: 50000)')
    parser.add_argument('--data-path', type=str, default="data/data.json.gz",
                       help='Path to the data file (default: data/data.json.gz)')
    parser.add_argument('--host', type=str, default="http://localhost:9308",
                       help='Manticore Search host (default: http://localhost:9308)')
    
    args = parser.parse_args()
    
    # Initialize comparator
    comparator = ManticoreComparator(
        dimension=1024,
        host=args.host,
        hnsw_m=16,
        ef_construction=200,
        ef_search=2000
    )
    comparator.query_log_path = "manticore_replace_queries.sql"
    with open(comparator.query_log_path, "w"):
        pass
    comparator.select_log_path = "manticore_select_queries.sql"
    with open(comparator.select_log_path, "w"):
        pass
    comparator.ddl_log_path = "manticore_ddl_queries.sql"
    with open(comparator.ddl_log_path, "w"):
        pass
    
    # Load data
    comparator.load_data(args.data_path, max_rows=args.max_rows, rebuild_index=args.rebuild)
    print("\nManticore index is ready for queries!")
    print("Call comparator.search(query_vector, k=50) to perform a search.")

    # Example: You can add a query here later
    query_text = f"""Synthesis and Structural Characterization of Alkyl, Gallium-Phosphorus Compounds, X-Ray Crystal Structures of (Me3CCH2)2(Cl)Ga.P(SiMe3)3, R2GaP(SiMe3)2GaR2Cl(R=Me3CCH2 and Me3SiCH2), 
    and ((R)(X)GaP(SiMe3)2)2 (R=Me3CCH2, X=Cl; R=Me3CCH2, X=Me3CCH2; R=Me3SiCH2,X=Br)"""
    query_vector = [0.1665349155664444,0.3510439395904541,-0.6301337480545044,0.02158646658062935,0.6581355333328247,0.09361449629068375,-0.1667909324169159,-0.3827958106994629,0.041684236377477646,0.01769174262881279,0.22897650301456451,0.2472117692232132,-1.7658517360687256,-0.7131491899490356,1.2799150943756104,1.1499123573303223,-0.7476005554199219,0.33797550201416016,-0.4572656750679016,0.2690719664096832,1.8523008823394775,-0.14790426194667816,0.6064537167549133,0.33932507038116455,0.02048667147755623,0.5762794613838196,-0.598640501499176,0.2992035448551178,-0.10291647911071777,0.6211121082305908,-0.17155319452285767,-0.33324170112609863,0.1249954104423523,0.505691409111023,-0.7652907371520996,-0.5120964646339417,-0.5211066007614136,0.5758886933326721,0.20649665594100952,1.0987881422042847,1.182282567024231,0.8278077244758606,0.3038588762283325,2.0491819381713867,-0.5734562873840332,-0.10507280379533768,-1.362227439880371,-0.059989623725414276,-0.13140684366226196,0.7520959377288818,-0.07011072337627411,-0.18782325088977814,-0.32736897468566895,0.5471047163009644,0.18469880521297455,-0.05823170393705368,0.9159368276596069,-0.25906771421432495,-0.5835723876953125,0.264801949262619,-0.706909716129303,0.15052197873592377,0.02147914469242096,-0.11432072520256042,0.07316473871469498,-0.18613502383232117,0.6284405589103699,0.455153226852417,0.05542420595884323,-0.9723778367042542,-0.5220791697502136,1.4631327390670776,-1.677812099456787,-0.3548177182674408,-0.35128265619277954,0.12231620401144028,-0.23583441972732544,0.8731092214584351,0.25701451301574707,0.648453414440155,0.39831551909446716,-0.013046815991401672,0.45777130126953125,-0.2926371991634369,-0.06800107657909393,-0.4445902109146118,-0.23185233771800995,-0.17323081195354462,0.05829838663339615,-0.9140021800994873,-0.6400805115699768,-1.29897141456604,-0.13223358988761902,0.10406409949064255,0.29132920503616333,0.33687180280685425,-1.127565622329712,-0.3794131577014923,0.4422946572303772,0.37462174892425537,0.7567089796066284,-0.18913309276103973,0.011497633531689644,-0.06212818622589111,-0.04013828933238983,0.02675163745880127,0.8939313292503357,0.5787904262542725,-0.09064848721027374,0.286421537399292,0.8399891257286072,-0.7536532282829285,-0.1924751102924347,0.18520301580429077,-0.38906383514404297,-0.6210821270942688,0.7103552222251892,0.2580779492855072,0.27863913774490356,0.05725720524787903,-0.43409696221351624,-0.3840321898460388,0.7858403325080872,0.060479819774627686,0.2429261952638626,-0.012856332585215569,0.10505986213684082,0.34646013379096985,-0.28597843647003174,0.8945374488830566,1.6236250400543213,-0.19931915402412415,0.6735893487930298,-0.38120999932289124,1.254271149635315,-0.8417357206344604,0.18847449123859406,0.7737483978271484,-0.17054571211338043,-0.07482023537158966,0.07836395502090454,0.5574001669883728,0.9759781956672668,-0.7086277008056641,0.8776814937591553,0.6942659616470337,1.134427547454834,0.4308623969554901,0.8137772679328918,1.1862765550613403,-0.3452496826648712,-0.6359670758247375,-1.1165482997894287,0.390176922082901,-1.2054349184036255,0.36924803256988525,0.6189819574356079,-0.3061656653881073,-0.748856246471405,-0.9806089401245117,0.7986326217651367,-0.14300163090229034,0.49265462160110474,-0.19270838797092438,0.02490546554327011,-0.46191632747650146,0.07177138328552246,0.6764244437217712,-0.1436905860900879,-0.18784821033477783,0.2248932421207428,0.06701666861772537,0.15634691715240479,-0.026495005935430527,1.1731435060501099,-0.021081293001770973,0.0837024450302124,0.7382928133010864,0.1775541603565216,0.1923229992389679,-0.6371700167655945,0.41455531120300293,-0.36900031566619873,-0.6130245327949524,-0.15900647640228271,-0.006744276732206345,-0.31427404284477234,-0.22675110399723053,-0.9853413701057434,0.3323791027069092,0.7270920872688293,-0.3381958305835724,0.08742381632328033,-0.22051125764846802,1.0343934297561646,-0.5262178778648376,-0.013550244271755219,0.5861223340034485,-0.6934283375740051,0.17061197757720947,0.01821817457675934,-0.2075788378715515,0.25120338797569275,0.703406572341919,-0.8012257814407349,-1.0321851968765259,0.3581380844116211,-0.4876441955566406,-1.1993883848190308,-0.20094913244247437,-0.7750324606895447,-0.3335544466972351,0.2697612941265106,1.2149876356124878,0.2791685163974762,-0.061013929545879364,-0.3223521411418915,0.1826891005039215,-0.22064919769763947,0.3096678853034973,-0.9834625720977783,0.4607226252555847,0.5526198744773865,0.4218974709510803,0.08804649859666824,-0.31676971912384033,-0.2690424621105194,-0.4505186975002289,0.689865231513977,0.4889810383319855,0.5067600607872009,-0.24079804122447968,-0.9746337532997131,0.5815855860710144,0.9928292632102966,0.49188169836997986,-0.43791520595550537,0.4415613114833832,0.3681797981262207,-0.3759010136127472,0.15978161990642548,-0.8156440258026123,-0.040598347783088684,0.1106976866722107,-0.18076643347740173,-0.3770049810409546,0.4324772357940674,-1.683651089668274,0.6902864575386047,-0.34620383381843567,0.33496254682540894,-1.1864702701568604,-0.9575545787811279,-0.5162032842636108,0.013594454154372215,0.07012876868247986,-0.08237304538488388,-0.04878591001033783,-1.1922158002853394,1.2085946798324585,0.26315176486968994,0.6258333325386047,-0.9507230520248413,0.408317506313324,1.7450315952301025,-0.9396291971206665,-0.6532787084579468,-0.05963373929262161,0.14909812808036804,-0.8151771426200867,-0.11808105558156967,0.8248403072357178,0.5175917148590088,-0.12263478338718414,0.11880000680685043,-0.534372091293335,0.4291084110736847,0.7443824410438538,-1.045760154724121,-1.0357826948165894,-0.3641587793827057,1.1191328763961792,0.301065057516098,-0.7613474726676941,0.19176170229911804,0.5573378801345825,-0.32206428050994873,-0.7679091095924377,0.5382357835769653,-0.05333336442708969,0.033179692924022675,-1.4979006052017212,-0.2067829817533493,0.034413449466228485,-0.39430245757102966,-0.8578941226005554,-0.12997877597808838,0.792827844619751,0.5679923892021179,-0.17669329047203064,0.5387427806854248,-0.5340059995651245,-0.5616418719291687,-0.2138671576976776,0.23810696601867676,0.39715659618377686,0.5643587708473206,-0.0354020819067955,-0.1235283762216568,0.40794873237609863,0.8133578896522522,0.8146174550056458,0.3209444284439087,-0.2433399111032486,0.16307967901229858,-0.8353511095046997,0.22107921540737152,0.14099912345409393,0.25479036569595337,0.10014547407627106,-0.2533275783061981,0.2743341326713562,0.951264500617981,-0.3083832859992981,0.660097062587738,0.475826621055603,-0.5331376194953918,-0.6822001934051514,0.16981200873851776,0.230063796043396,-0.3875577747821808,-1.0255993604660034,-0.5217925310134888,-0.30435705184936523,1.0083520412445068,-0.5294346213340759,0.007369827479124069,-0.15148188173770905,0.08224991708993912,0.3125234842300415,-0.21884803473949432,-0.3424997925758362,-0.19412924349308014,1.039854884147644,1.6381635665893555,-0.13264018297195435,0.21563182771205902,0.15768691897392273,-0.09241672605276108,-1.0017379522323608,-0.20878945291042328,-0.16822916269302368,-0.10395927727222443,0.6548712849617004,0.9132037162780762,-0.2578284442424774,1.078141450881958,0.011083010584115982,-0.05755139887332916,0.5679713487625122,-0.3302822709083557,0.2644665241241455,0.8560774922370911,-0.060548800975084305,0.2987092137336731,-0.730270266532898,-0.25014370679855347,-0.7476979494094849,0.334237277507782,0.8932916522026062,-0.07384205609560013,-0.22590260207653046,-0.4045763313770294,0.6885961890220642,0.08326909691095352,-0.09361402690410614,-0.653398334980011,-0.5618414282798767,-0.4751614034175873,0.3400414288043976,0.4130353629589081,0.9771044254302979,-1.0760835409164429,-0.48051512241363525,-0.36404791474342346,-0.7307525277137756,-0.4069482088088989,-0.24781352281570435,-0.552322506904602,0.9813048243522644,-0.014024856500327587,-0.06993160396814346,-0.0007446333765983582,0.7567121386528015,0.6986557841300964,0.06328213214874268,-0.011300312355160713,-0.10181143879890442,0.5821546912193298,0.4243308901786804,-1.1049848794937134,-0.7330402135848999,-0.9286959171295166,1.4077973365783691,-0.24705584347248077,0.080302894115448,-0.35071954131126404,0.22834543883800507,-0.5225294828414917,0.22384591400623322,0.054189153015613556,-0.07247243821620941,0.33549991250038147,-0.03513592481613159,0.2793586552143097,0.11999817937612534,-0.08597303926944733,-0.6520525813102722,-0.46030932664871216,-0.17546813189983368,0.05389970913529396,0.3890593349933624,0.1946994662284851,-0.07880383729934692,0.1377897560596466,0.20986807346343994,0.28271016478538513,0.4951786398887634,0.39649975299835205,-0.26416879892349243,-0.41532275080680847,0.807643473148346,0.13566918671131134,0.4540201723575592,0.31428560614585876,0.16593395173549652,-0.9567309617996216,0.07133492827415466,-0.34185850620269775,0.42921334505081177,-0.7698834538459778,-0.2110942006111145,-0.1753503978252411,-0.5267536640167236,-0.08333810418844223,0.2733839750289917,0.5153862833976746,-0.266741544008255,0.3867720365524292,-0.5322685241699219,0.05630578100681305,-0.023406531661748886,0.9838622212409973,1.061934471130371,0.10408716648817062,-0.20277629792690277,0.21512272953987122,-0.14150337874889374,-0.9471353888511658,-1.0052367448806763,-0.6075102686882019,-0.06133800372481346,-0.3860104978084564,-0.10946130752563477,-0.023060711100697517,0.6482081413269043,1.0111690759658813,-0.7817526459693909,1.5301989316940308,0.2669985592365265,0.9352572560310364,0.004027392715215683,-0.33551621437072754,0.04202932119369507,0.8856926560401917,-0.5376384854316711,-0.07803000509738922,0.43093496561050415,0.2676360011100769,-0.1744275838136673,1.3579509258270264,0.26793503761291504,-0.07297837734222412,0.9240302443504333,-0.11583879590034485,-0.1148928552865982,-0.15399141609668732,-1.23020339012146,-0.4168378412723541,1.0158116817474365,0.5446805357933044,0.5497846007347107,-0.4292181134223938,-0.5527447462081909,0.42707881331443787,0.5080228447914124,0.10520200431346893,-0.23271676898002625,-0.11565997451543808,0.4177948236465454,-1.8999656438827515,-0.4473520517349243,-0.07244917750358582,-0.09010198712348938,-0.6386889815330505,-0.13753466308116913,-0.40996840596199036,-0.35607773065567017,-1.0843689441680908,0.9095877408981323,-1.4547029733657837,0.9071512818336487,-0.3309633731842041,-1.037647008895874,-0.5526483058929443,-0.22250036895275116,0.5603821873664856,-0.3077637553215027,-1.050326943397522,0.10573767125606537,-0.3270609676837921,0.37748414278030396,0.421668142080307,-0.38266485929489136,0.8136051893234253,-0.01677352376282215,-0.19512173533439636,0.15467384457588196,0.23261314630508423,-0.2561533451080322,-0.6836565136909485,-0.34694960713386536,-0.20543082058429718,0.6004523634910583,-0.13091138005256653,0.4836272597312927,0.2921900153160095,-0.4548017084598541,-0.12608948349952698,0.1370866596698761,0.41717755794525146,0.1522696316242218,0.16777244210243225,0.2011612057685852,0.708328902721405,0.7614238262176514,-0.7117117047309875,0.4746887683868408,-0.6894504427909851,-0.9543516635894775,0.6218982338905334,-0.6221557855606079,1.288405418395996,0.6884045600891113,-0.8834354877471924,-0.42819634079933167,-0.5341002345085144,0.004321053624153137,0.025853797793388367,1.263287901878357,0.571872353553772,-0.8769412040710449,-0.7887914776802063,0.6017126441001892,-1.103920340538025,1.0108473300933838,0.23285552859306335,-0.6319761276245117,-0.815516471862793,0.21977856755256653,-0.7527555227279663,-0.06925775110721588,0.4865870475769043,-1.1618316173553467,-0.5282827615737915,0.18324986100196838,0.8322747349739075,0.3524310886859894,0.13800419867038727,1.2923153638839722,0.4146976172924042,0.20593003928661346,0.673662006855011,-0.23492877185344696,-0.5862026214599609,0.27377811074256897,0.1165751963853836,0.1712087094783783,-0.19239160418510437,0.29328322410583496,-1.0457346439361572,0.21405769884586334,0.8347033262252808,-0.265998512506485,0.41723519563674927,0.7963252663612366,0.05389423668384552,-0.6817244291305542,-0.4774218797683716,0.0793195590376854,-0.17406171560287476,-0.10488982498645782,0.325320839881897,-0.34707099199295044,-0.8645579814910889,-0.0827299952507019,0.25811392068862915,-0.19251279532909393,0.3238959312438965,-0.17498645186424255,-1.2041857242584229,-0.3684978485107422,0.1453561782836914,-0.5005611181259155,1.3716884851455688,-0.4723253846168518,0.09929797798395157,1.5085973739624023,-0.4252298176288605,1.045859694480896,0.7894294857978821,-0.3102857172489166,0.6735650300979614,-1.1224653720855713,0.9204750061035156,0.5124058723449707,-0.6428247094154358,-0.5589251518249512,-0.8137417435646057,-0.1715051233768463,-0.6752670407295227,-0.13359741866588593,-0.8421294093132019,-0.4895702004432678,-0.8975744247436523,-0.1315302848815918,-1.6228631734848022,-1.811037540435791,0.45094209909439087,0.21104219555854797,0.21882794797420502,0.728706955909729,0.9487420916557312,0.16271480917930603,0.5940768718719482,-0.46708548069000244,0.03451843187212944,-0.3861137926578522,-0.34928351640701294,-0.24025236070156097,0.3084394335746765,0.33344966173171997,-0.054103653877973557,0.8693914413452148,0.7460733652114868,-0.36006197333335876,-0.36075204610824585,-0.338029682636261,-0.029984407126903534,0.27215299010276794,-0.3315656781196594,0.019463051110506058,-0.02885984629392624,0.8508500456809998,0.20866486430168152,0.25518909096717834,-0.25132879614830017,0.5201305150985718,-0.7117713689804077,-1.1775989532470703,0.6223030090332031,-0.35645121335983276,-0.5864695310592651,-0.467921644449234,0.6176456809043884,-0.16462677717208862,-0.38079071044921875,0.7877861261367798,0.37850916385650635,0.09525705128908157,0.5417521595954895,-0.7359251976013184,0.4528399407863617,-0.557715654373169,0.09840750694274902,-0.7378807663917542,-0.11889694631099701,-0.30059880018234253,-0.27775558829307556,0.29113227128982544,0.8091611862182617,0.21503601968288422,-0.7505373358726501,0.9184859991073608,-0.7478613257408142,-0.7734431624412537,0.47028663754463196,0.9375384449958801,0.2952231764793396,-0.5997596979141235,-1.0579246282577515,0.7693190574645996,-0.3357221186161041,0.005883406847715378,0.030932743102312088,-0.8867594599723816,0.7389141321182251,-0.04057690501213074,0.6094115376472473,-0.5514024496078491,0.7431601881980896,-0.1515466570854187,-1.4162523746490479,1.124887228012085,0.48307254910469055,-0.3140576481819153,0.02420111931860447,0.5462729334831238,-0.8898526430130005,-0.12202755361795425,-0.9830279350280762,0.3956961929798126,-1.1289900541305542,-0.7711795568466187,-1.0536270141601562,0.8151769042015076,-0.0003762589767575264,0.0817803144454956,-0.37353628873825073,0.166848286986351,-0.02316620759665966,-0.877474308013916,0.2885458171367645,-0.6163684725761414,1.0176098346710205,-0.08760949224233627,-0.2855307459831238,0.34752070903778076,0.552196741104126,-1.023153305053711,0.7183036804199219,-0.15128351747989655,0.9369897246360779,-0.19072656333446503,-0.16491475701332092,0.9604038000106812,0.5675939321517944,-0.8815714716911316,1.9405800104141235,0.6886951327323914,0.018469519913196564,-0.5234344601631165,0.49956417083740234,0.05124114081263542,-1.0846502780914307,-1.1920180320739746,-0.4307054579257965,-0.5816632509231567,-0.034686487168073654,-0.07760303467512131,0.24748334288597107,-0.4219092130661011,1.0197322368621826,-0.9615930318832397,0.07198095321655273,-0.6773454546928406,0.42440474033355713,0.3401220440864563,-0.9511722326278687,-0.5579714179039001,0.28688061237335205,0.36472442746162415,-0.16440829634666443,0.7989516258239746,-0.2529507577419281,-0.7147117853164673,-0.03780468553304672,-0.32406309247016907,-0.5879298448562622,-1.2949801683425903,0.6916909217834473,1.0979292392730713,-0.25096428394317627,-0.8359824419021606,0.07528641819953918,0.4961021840572357,0.1181693822145462,-0.1612360030412674,-0.4075978994369507,-0.05369764193892479,-0.29501593112945557,0.4894388020038605,0.5444069504737854,-0.34563520550727844,0.2650757431983948,-0.27338671684265137,0.6163816452026367,-1.2601802349090576,0.7252885103225708,0.2916334271430969,0.8697890043258667,-0.3621423542499542,-0.17868947982788086,-0.3197164833545685,-0.02468237280845642,-0.2927044630050659,-0.2494649589061737,0.13044872879981995,0.3133104145526886,0.7660455107688904,-0.203252375125885,0.479748398065567,-0.6574370861053467,-0.5530492663383484,-0.752343475818634,-0.08191575855016708,-0.15147151052951813,-0.9633767604827881,-0.014943723566830158,0.11056811362504959,-0.047890547662973404,-0.8127325177192688,0.2969043552875519,0.2723250985145569,-0.7065584063529968,-0.44150790572166443,0.0380692221224308,-0.6300865411758423,0.08208616077899933,-1.2564468383789062,0.4247722327709198,0.47698721289634705,0.13483066856861115,0.6309182047843933,-0.8701720237731934,0.820073664188385,-0.5437712669372559,0.2848956286907196,-0.7949746251106262,0.8388857841491699,0.9807114005088806,-1.1484416723251343,0.10012724995613098,0.014935377985239029,0.24206703901290894,-0.852995753288269,-0.7486050128936768,0.3199211061000824,0.9012508988380432,-0.5624381899833679,-0.1126609593629837,-0.5327825546264648,-0.2846008539199829,0.5662137866020203,-0.14206410944461823,-0.407953679561615,-0.35240939259529114,0.28322839736938477,-0.09329089522361755,-0.014478079974651337,0.4937896132469177,-0.39730408787727356,0.3323308825492859,-0.2600560486316681,-0.35364559292793274,1.5365068912506104,-0.6921931505203247,-0.24606192111968994,1.108777642250061,0.05014358460903168,0.15075810253620148,-0.25678181648254395,-0.18222835659980774,0.27905580401420593,0.8226780891418457,-0.2521129250526428,-0.3328794240951538,-0.007515748962759972,0.07582955062389374,0.20725512504577637,0.8105095624923706,-0.05971285700798035,0.5673227310180664,0.27277064323425293,0.7246171832084656,-0.46691155433654785,0.2022264301776886,-0.3915267288684845,0.14168962836265564,-0.4276314675807953,-0.3811224699020386,1.2046536207199097,-0.5138994455337524,0.13333183526992798,0.5253404378890991,-1.358442783355713,-0.12128172814846039,0.007681500166654587,-0.06414248794317245,-1.1444554328918457,0.38923829793930054,-0.17362022399902344,-0.07068119198083878,1.132704734802246,0.7092016339302063,0.15280580520629883,-0.22737979888916016,0.24117273092269897,0.30383920669555664,-0.14207521080970764,-0.4647881090641022,0.5428688526153564,0.2619404196739197,0.39581143856048584,-0.29123395681381226,-0.06170639395713806,0.7944145798683167,0.6093540787696838,-0.23434394598007202,0.3842809796333313,0.5575763583183289,0.4421333372592926,0.6306602358818054,-0.08644159138202667,-0.4155883193016052,0.5509064793586731,0.3844458758831024,0.21130137145519257,0.005605883896350861,0.04284362494945526,-0.5750231146812439,-0.17030495405197144,0.06377197802066803,-0.770916759967804,-0.5321249961853027,-0.20919114351272583,0.1631956845521927,1.1169427633285522,0.5684211850166321,-0.14658284187316895,0.3399352431297302,-0.14132168889045715,-0.1497449427843094,0.8222185969352722,-0.2568095922470093,0.34365200996398926,-0.16067376732826233,0.08770176768302917,-0.9935805797576904,-0.229875847697258,-0.5901567935943604,0.1261761337518692,-0.03410276398062706,0.12483718246221542,-0.07273925095796585,-0.27557432651519775,0.7034529447555542,0.9452341198921204,-0.24081875383853912,0.3061710596084595,-0.9606319665908813,-0.15421722829341888,0.4366954267024994,0.6574212312698364,0.997888445854187,0.19719702005386353,-0.06342341005802155,0.21240444481372833,-0.20062625408172607,0.01978949084877968,-0.9584136009216309,-0.747623085975647,0.23070935904979706,0.007588332053273916,0.2572326064109802,0.3450084924697876,0.23808540403842926,0.4538947641849518,-0.7158849835395813,0.45559772849082947,0.7678901553153992,2.292950391769409,-0.24750711023807526,0.8160589337348938,-0.03828756511211395,-1.0801959037780762,0.07921621948480606,0.16782985627651215,-0.22000528872013092,-0.4499518871307373,-0.5094490051269531,0.09910847991704941,0.1973029524087906,-0.24977782368659973,0.7899564504623413,-0.5543602705001831,-0.23296678066253662,0.21235325932502747,0.7764322757720947,-1.5614244937896729,-0.7757338285446167,0.23124924302101135,-0.4859873056411743,0.5832569599151611,-0.012084726244211197,-0.49915096163749695,0.3547685742378235,-0.3613446354866028,-0.59247887134552,0.17599686980247498,-0.5798702239990234,-0.5949311256408691,-0.46360450983047485,0.056494519114494324,0.7828671336174011,1.135758876800537,-0.8823466897010803,0.41567474603652954,1.245957851409912,0.20890364050865173,-0.5496659874916077,-0.5113167762756348,-0.8470510244369507,-1.0006924867630005]  # 1024 dimensions
    print("--------------------------------")
    print("Query: ", query_text)
    print("--------------------------------")
    comparator.search_with_kb(query_vector, k=50)


if __name__ == "__main__":
    main()
