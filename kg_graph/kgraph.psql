DROP TABLE entities, relationships, fact_tuples;

-- Table for storing entities
CREATE TABLE entities (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(255) NOT NULL -- 'PER', 'OBJ', 'LOC'
    -- properties VARCHAR(1000) NOT NULL
);

-- Table for storing relationships
CREATE TABLE relationships (
    id SERIAL PRIMARY KEY,
    relationship_type VARCHAR(255) -- e.g., 'Produces', 'Mentioned'
);

-- Table for storing facts (relationships between entities)
CREATE TABLE fact_tuples (
    id SERIAL PRIMARY KEY,
    source_entity_id INTEGER,
    relationship_id INTEGER, -- Foreign key to relationships table
    target_entity_id INTEGER,
    created_at timestamp default CURRENT_TIMESTAMP,
    recall_strength FLOAT DEFAULT 1 CHECK (recall_strength >= 0 AND recall_strength <= 1),
    agent_uuid                    uuid,
    FOREIGN KEY (source_entity_id) REFERENCES entities(id),
    FOREIGN KEY (relationship_id) REFERENCES relationships(id),
    FOREIGN KEY (target_entity_id) REFERENCES entities(id)
);

