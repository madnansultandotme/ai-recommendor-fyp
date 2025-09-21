import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from sentence_transformers import SentenceTransformer
import uuid
import json

from app.config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """Vector database manager using ChromaDB for semantic search."""
    
    def __init__(self):
        self.client = None
        self.embedding_function = None
        self.sentence_model = None
        self.collections = {}
        self.setup_vector_db()
    
    def setup_vector_db(self):
        """Setup ChromaDB client and embedding function."""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=settings.chroma_persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=settings.enable_telemetry,
                    allow_reset=True
                )
            )
            
            # Initialize sentence transformer model
            self.sentence_model = SentenceTransformer(
                settings.sentence_transformer_model,
                cache_folder=settings.huggingface_cache_dir
            )
            
            # Setup embedding function
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.sentence_transformer_model,
                cache_folder=settings.huggingface_cache_dir
            )
            
            # Initialize collections
            self._setup_collections()
            
            logger.info("Vector database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup vector database: {e}")
            raise
    
    def _setup_collections(self):
        """Setup ChromaDB collections for different data types."""
        collection_configs = {
            "startups": {
                "metadata": {"description": "Startup descriptions and metadata"}
            },
            "positions": {
                "metadata": {"description": "Job positions and requirements"}
            },
            "user_skills": {
                "metadata": {"description": "User skills and preferences"}
            }
        }
        
        for collection_name, config in collection_configs.items():
            try:
                collection = self.client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function,
                    metadata=config.get("metadata", {})
                )
                self.collections[collection_name] = collection
                logger.info(f"Collection '{collection_name}' initialized")
                
            except Exception as e:
                logger.error(f"Failed to setup collection '{collection_name}': {e}")
                raise
    
    def add_startup_embedding(
        self, 
        startup_id: int, 
        description: str, 
        metadata: Dict[str, Any]
    ) -> str:
        """Add startup embedding to vector store."""
        try:
            doc_id = f"startup_{startup_id}"
            
            # Combine description with relevant metadata for better embeddings
            combined_text = self._combine_startup_text(description, metadata)
            
            self.collections["startups"].add(
                documents=[combined_text],
                metadatas=[{
                    "startup_id": startup_id,
                    "type": "startup",
                    **metadata
                }],
                ids=[doc_id]
            )
            
            logger.debug(f"Added startup embedding for ID: {startup_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to add startup embedding: {e}")
            raise
    
    def add_position_embedding(
        self,
        position_id: int,
        title: str,
        description: str,
        requirements: Dict[str, Any],
        startup_metadata: Dict[str, Any] = None
    ) -> str:
        """Add position embedding to vector store."""
        try:
            doc_id = f"position_{position_id}"
            
            # Combine all text fields for better embeddings
            combined_text = self._combine_position_text(
                title, description, requirements, startup_metadata
            )
            
            metadata = {
                "position_id": position_id,
                "type": "position",
                "title": title,
                **requirements,
            }
            
            if startup_metadata:
                metadata.update(startup_metadata)
            
            self.collections["positions"].add(
                documents=[combined_text],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            logger.debug(f"Added position embedding for ID: {position_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to add position embedding: {e}")
            raise
    
    def add_user_skills_embedding(
        self,
        user_id: int,
        skills: List[str],
        preferences: Dict[str, Any]
    ) -> str:
        """Add user skills embedding to vector store."""
        try:
            doc_id = f"user_{user_id}"
            
            # Combine skills and preferences
            combined_text = self._combine_user_text(skills, preferences)
            
            self.collections["user_skills"].add(
                documents=[combined_text],
                metadatas=[{
                    "user_id": user_id,
                    "type": "user_skills",
                    "skills": skills,
                    **preferences
                }],
                ids=[doc_id]
            )
            
            logger.debug(f"Added user skills embedding for ID: {user_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to add user skills embedding: {e}")
            raise
    
    def search_similar_startups(
        self,
        query_text: str,
        n_results: int = 10,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar startups based on query text."""
        try:
            where_clause = {"type": "startup"}
            if filters:
                where_clause.update(filters)
            
            results = self.collections["startups"].query(
                query_texts=[query_text],
                n_results=n_results,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            return self._format_search_results(results)
            
        except Exception as e:
            logger.error(f"Failed to search similar startups: {e}")
            return []
    
    def search_similar_positions(
        self,
        user_skills: List[str],
        preferences: Dict[str, Any] = None,
        n_results: int = 10,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar positions based on user skills."""
        try:
            # Create query from user skills and preferences
            query_text = self._combine_user_text(user_skills, preferences or {})
            
            where_clause = {"type": "position"}
            if filters:
                where_clause.update(filters)
            
            results = self.collections["positions"].query(
                query_texts=[query_text],
                n_results=n_results,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            return self._format_search_results(results)
            
        except Exception as e:
            logger.error(f"Failed to search similar positions: {e}")
            return []
    
    def search_similar_users(
        self,
        query_text: str,
        n_results: int = 10,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Search for users with similar skills."""
        try:
            where_clause = {"type": "user_skills"}
            if filters:
                where_clause.update(filters)
            
            results = self.collections["user_skills"].query(
                query_texts=[query_text],
                n_results=n_results,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            return self._format_search_results(results)
            
        except Exception as e:
            logger.error(f"Failed to search similar users: {e}")
            return []
    
    def update_embedding(
        self,
        collection_name: str,
        doc_id: str,
        document: str,
        metadata: Dict[str, Any]
    ):
        """Update an existing embedding."""
        try:
            if collection_name not in self.collections:
                raise ValueError(f"Collection '{collection_name}' not found")
            
            self.collections[collection_name].update(
                ids=[doc_id],
                documents=[document],
                metadatas=[metadata]
            )
            
            logger.debug(f"Updated embedding {doc_id} in collection {collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to update embedding: {e}")
            raise
    
    def delete_embedding(self, collection_name: str, doc_id: str):
        """Delete an embedding."""
        try:
            if collection_name not in self.collections:
                raise ValueError(f"Collection '{collection_name}' not found")
            
            self.collections[collection_name].delete(ids=[doc_id])
            logger.debug(f"Deleted embedding {doc_id} from collection {collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to delete embedding: {e}")
            raise
    
    def get_collection_count(self, collection_name: str) -> int:
        """Get count of documents in a collection."""
        try:
            if collection_name not in self.collections:
                return 0
            
            return self.collections[collection_name].count()
            
        except Exception as e:
            logger.error(f"Failed to get collection count: {e}")
            return 0
    
    def reset_collection(self, collection_name: str):
        """Reset a collection by deleting all documents."""
        try:
            if collection_name not in self.collections:
                raise ValueError(f"Collection '{collection_name}' not found")
            
            self.client.delete_collection(collection_name)
            
            # Recreate the collection
            collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            self.collections[collection_name] = collection
            
            logger.info(f"Reset collection '{collection_name}'")
            
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            raise
    
    def _combine_startup_text(self, description: str, metadata: Dict[str, Any]) -> str:
        """Combine startup description with metadata for better embeddings."""
        text_parts = [description]
        
        if "industry" in metadata:
            industries = metadata["industry"] if isinstance(metadata["industry"], list) else [metadata["industry"]]
            text_parts.append(f"Industry: {', '.join(industries)}")
        
        if "stage" in metadata:
            text_parts.append(f"Stage: {metadata['stage']}")
        
        if "tags" in metadata:
            tags = metadata["tags"] if isinstance(metadata["tags"], list) else [metadata["tags"]]
            text_parts.append(f"Tags: {', '.join(tags)}")
        
        if "location" in metadata:
            text_parts.append(f"Location: {metadata['location']}")
        
        return " ".join(text_parts)
    
    def _combine_position_text(
        self,
        title: str,
        description: str,
        requirements: Dict[str, Any],
        startup_metadata: Dict[str, Any] = None
    ) -> str:
        """Combine position information for better embeddings."""
        text_parts = [f"Position: {title}", description]
        
        if "required_skills" in requirements:
            skills = requirements["required_skills"]
            if isinstance(skills, list):
                text_parts.append(f"Required skills: {', '.join(skills)}")
        
        if "preferred_skills" in requirements:
            skills = requirements["preferred_skills"]
            if isinstance(skills, list):
                text_parts.append(f"Preferred skills: {', '.join(skills)}")
        
        if startup_metadata:
            if "industry" in startup_metadata:
                industries = startup_metadata["industry"] if isinstance(startup_metadata["industry"], list) else [startup_metadata["industry"]]
                text_parts.append(f"Company industry: {', '.join(industries)}")
        
        return " ".join(text_parts)
    
    def _combine_user_text(self, skills: List[str], preferences: Dict[str, Any]) -> str:
        """Combine user skills and preferences for better embeddings."""
        text_parts = [f"Skills: {', '.join(skills)}"]
        
        if "preferred_industries" in preferences:
            industries = preferences["preferred_industries"]
            if isinstance(industries, list):
                text_parts.append(f"Preferred industries: {', '.join(industries)}")
        
        if "preferred_roles" in preferences:
            roles = preferences["preferred_roles"]
            if isinstance(roles, list):
                text_parts.append(f"Preferred roles: {', '.join(roles)}")
        
        return " ".join(text_parts)
    
    def _format_search_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format ChromaDB search results."""
        formatted_results = []
        
        if not results['documents'][0]:  # No results
            return formatted_results
        
        for i in range(len(results['documents'][0])):
            result = {
                "document": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "similarity_score": 1 - results['distances'][0][i],  # Convert distance to similarity
            }
            formatted_results.append(result)
        
        return formatted_results


# Global vector store instance
vector_store = VectorStore()