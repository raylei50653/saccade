-- Saccade Ablation Test Data Schema

CREATE TABLE IF NOT EXISTS experiments (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    config JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
    id SERIAL PRIMARY KEY,
    experiment_id INTEGER REFERENCES experiments(id) ON DELETE CASCADE,
    dataset TEXT NOT NULL, -- e.g., 'MOT17-02-FRCNN'
    idf1 FLOAT,
    recall FLOAT,
    precision FLOAT,
    mota FLOAT,
    num_switches INTEGER,
    num_false_positives INTEGER,
    num_misses INTEGER,
    raw_results JSONB, -- Store full output if needed
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_experiments_category ON experiments(category);
CREATE INDEX idx_metrics_experiment_id ON metrics(experiment_id);
