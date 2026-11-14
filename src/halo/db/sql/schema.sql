CREATE TABLE public.molecules (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  user_id uuid NOT NULL REFERENCES auth.users (id) ON DELETE CASCADE,
  canonical_smiles text NOT NULL,
  standardization_version text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (user_id, canonical_smiles),
  UNIQUE (id, user_id)
);

CREATE TABLE public.molecule_profiles (
  molecule_id bigint PRIMARY KEY REFERENCES public.molecules (id) ON DELETE CASCADE,
  molecular_weight_g_mol double precision NOT NULL CHECK (molecular_weight_g_mol > 0),
  logp double precision NOT NULL,
  tpsa_angstrom_sq double precision NOT NULL CHECK (tpsa_angstrom_sq >= 0),
  heavy_atom_count integer NOT NULL CHECK (heavy_atom_count > 0),
  h_bond_acceptors integer NOT NULL CHECK (h_bond_acceptors >= 0),
  h_bond_donors integer NOT NULL CHECK (h_bond_donors >= 0),
  rotatable_bonds integer NOT NULL CHECK (rotatable_bonds >= 0),
  ring_count integer NOT NULL CHECK (ring_count >= 0),
  aromatic_rings integer NOT NULL CHECK (aromatic_rings >= 0),
  qed double precision NOT NULL CHECK (qed BETWEEN 0 AND 1),
  lipinski_violations smallint NOT NULL CHECK (lipinski_violations BETWEEN 0 AND 4),
  sa_score jsonb NOT NULL CHECK (jsonb_typeof(sa_score) = 'object'),
  pains_alerts text[] NOT NULL,
  predictions jsonb NOT NULL CHECK (jsonb_typeof(predictions) = 'object'),
  computed_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE public.projects (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users (id) ON DELETE CASCADE,
  source_molecule_id bigint NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (id, user_id),
  UNIQUE (user_id, source_molecule_id),
  FOREIGN KEY (source_molecule_id, user_id) REFERENCES public.molecules (id, user_id)
);

CREATE TABLE public.sessions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id uuid NOT NULL REFERENCES public.projects (id) ON DELETE CASCADE,
  state jsonb NOT NULL CHECK (jsonb_typeof(state) = 'object'),
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (id, project_id)
);

CREATE INDEX ON public.sessions (project_id);

CREATE TABLE public.turns (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  session_id uuid NOT NULL,
  project_id uuid NOT NULL,
  turn_index integer NOT NULL CHECK (turn_index > 0),
  request_id uuid,
  request jsonb NOT NULL CHECK (jsonb_typeof(request) = 'object'),
  status text NOT NULL CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
  task text CHECK (task IN (
    'profile', 'retrieve', 'design', 'cluster', 'plot', 'refine_plot', 'select', 'strategy',
    'decide', 'remember', 'converge', 'activity', 'assay_results', 'recall', 'clarify'
  )),
  output jsonb CHECK (jsonb_typeof(output) = 'object'),
  trace jsonb CHECK (jsonb_typeof(trace) = 'object'),
  started_at timestamptz NOT NULL DEFAULT now(),
  finished_at timestamptz,
  UNIQUE (id, project_id),
  UNIQUE (session_id, turn_index),
  UNIQUE (session_id, request_id),
  FOREIGN KEY (session_id, project_id)
    REFERENCES public.sessions (id, project_id) ON DELETE CASCADE,
  CONSTRAINT turns_running_check CHECK (
    (status = 'running') = (finished_at IS NULL)
    AND (status = 'running') = (output IS NULL)
    AND (status = 'running') = (trace IS NULL)
  ),
  CONSTRAINT turns_task_status_check CHECK ((task IS NOT NULL) = (status = 'completed'))
);

CREATE UNIQUE INDEX turns_one_running_per_project ON public.turns (project_id)
WHERE status = 'running';
CREATE INDEX ON public.turns (project_id);

CREATE TABLE public.runs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users (id) ON DELETE CASCADE,
  source_molecule_id bigint NOT NULL,
  project_id uuid,
  parent_run_id uuid,
  turn_id bigint UNIQUE,
  status text NOT NULL CHECK (status IN ('running', 'completed', 'failed', 'limit_exhausted')),
  model text NOT NULL,
  config jsonb NOT NULL DEFAULT '{}' CHECK (jsonb_typeof(config) = 'object'),
  input_tokens integer NOT NULL DEFAULT 0 CHECK (input_tokens >= 0),
  output_tokens integer NOT NULL DEFAULT 0 CHECK (output_tokens >= 0),
  reasoning_tokens integer NOT NULL DEFAULT 0 CHECK (reasoning_tokens >= 0),
  cached_tokens integer NOT NULL DEFAULT 0 CHECK (cached_tokens >= 0),
  failure_reason text,
  request_id uuid,
  request jsonb CHECK (jsonb_typeof(request) = 'object'),
  created_at timestamptz NOT NULL DEFAULT now(),
  finished_at timestamptz,
  UNIQUE (id, project_id),
  FOREIGN KEY (source_molecule_id, user_id) REFERENCES public.molecules (id, user_id),
  FOREIGN KEY (project_id, user_id) REFERENCES public.projects (id, user_id) ON DELETE CASCADE,
  FOREIGN KEY (turn_id, project_id) REFERENCES public.turns (id, project_id),
  FOREIGN KEY (parent_run_id, project_id) REFERENCES public.runs (id, project_id),
  CONSTRAINT runs_finished_check CHECK ((status = 'running') = (finished_at IS NULL)),
  CONSTRAINT runs_failure_reason_check CHECK ((status = 'failed') = (failure_reason IS NOT NULL)),
  CONSTRAINT runs_project_check CHECK (
    (turn_id IS NULL) = (project_id IS NULL)
    AND (parent_run_id IS NULL OR project_id IS NOT NULL)
    AND (request_id IS NULL) = (request IS NULL)
    AND (request_id IS NULL OR project_id IS NULL)
  )
);

CREATE UNIQUE INDEX runs_user_request_key ON public.runs (user_id, request_id)
WHERE request_id IS NOT NULL;
CREATE INDEX ON public.runs (user_id, created_at DESC);
CREATE INDEX ON public.runs (project_id, created_at);
CREATE INDEX ON public.runs (source_molecule_id);
CREATE INDEX ON public.runs (parent_run_id);

CREATE TABLE public.candidates (
  run_id uuid NOT NULL,
  molecule_id bigint NOT NULL REFERENCES public.molecules (id),
  ordinal integer NOT NULL CHECK (ordinal >= 0),
  design jsonb CHECK (jsonb_typeof(design) = 'object'),
  predictions jsonb CHECK (jsonb_typeof(predictions) = 'object'),
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (run_id, molecule_id),
  UNIQUE (run_id, ordinal),
  CONSTRAINT candidates_run_fkey FOREIGN KEY (run_id) REFERENCES public.runs (id) ON DELETE CASCADE
);

CREATE INDEX ON public.candidates (molecule_id);

ALTER TABLE public.runs
  ADD CONSTRAINT runs_parent_candidate_fkey FOREIGN KEY (parent_run_id, source_molecule_id)
    REFERENCES public.candidates (run_id, molecule_id);

CREATE TABLE public.model_calls (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  turn_id bigint REFERENCES public.turns (id) ON DELETE CASCADE,
  run_id uuid REFERENCES public.runs (id) ON DELETE CASCADE,
  call_index integer NOT NULL CHECK (call_index > 0),
  agent_role text NOT NULL CHECK (
    agent_role IN ('orchestrator', 'researcher', 'optimizer', 'strategist', 'analyst')
  ),
  model text NOT NULL,
  status text NOT NULL CHECK (status IN ('completed', 'failed')),
  invocation_digest text NOT NULL,
  context_bytes integer NOT NULL CHECK (context_bytes >= 0),
  elapsed_ms double precision NOT NULL CHECK (elapsed_ms >= 0),
  usage jsonb CHECK (jsonb_typeof(usage) = 'object'),
  output jsonb CHECK (jsonb_typeof(output) = 'object'),
  request_json text,
  validation_status text CHECK (validation_status IN ('accepted', 'partial', 'rejected')),
  validation_issues jsonb CHECK (jsonb_typeof(validation_issues) = 'array'),
  UNIQUE (turn_id, call_index),
  UNIQUE (run_id, call_index),
  CONSTRAINT model_calls_parent_check CHECK (num_nonnulls(turn_id, run_id) = 1),
  CONSTRAINT model_calls_completed_output_check CHECK (
    (status = 'completed') = (output IS NOT NULL)
  ),
  CONSTRAINT model_calls_validation_check CHECK (
    (validation_status IS NULL) = (validation_issues IS NULL)
    AND (validation_status IS NULL OR status = 'completed')
  )
);

CREATE TABLE public.proposals (
  turn_id bigint NOT NULL REFERENCES public.turns (id) ON DELETE CASCADE,
  molecule_id bigint NOT NULL REFERENCES public.molecules (id),
  ordinal integer NOT NULL CHECK (ordinal >= 0),
  PRIMARY KEY (turn_id, molecule_id),
  UNIQUE (turn_id, ordinal)
);

CREATE INDEX ON public.proposals (molecule_id);

CREATE TABLE public.decisions (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  turn_id bigint NOT NULL UNIQUE,
  project_id uuid NOT NULL,
  decision_type text NOT NULL CHECK (decision_type IN ('accept_proposal', 'set_preference')),
  proposal_turn_id bigint,
  supersedes_id bigint UNIQUE,
  note text NOT NULL CHECK (length(btrim(note)) > 0),
  scope text,
  decided_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (id, project_id, decision_type),
  UNIQUE (id, project_id, scope),
  FOREIGN KEY (turn_id, project_id) REFERENCES public.turns (id, project_id) ON DELETE CASCADE,
  FOREIGN KEY (proposal_turn_id, project_id) REFERENCES public.turns (id, project_id),
  FOREIGN KEY (supersedes_id, project_id, decision_type)
    REFERENCES public.decisions (id, project_id, decision_type),
  CONSTRAINT decisions_type_fields_check CHECK (
    CASE decision_type
      WHEN 'accept_proposal' THEN proposal_turn_id IS NOT NULL AND scope IS NULL
      ELSE proposal_turn_id IS NULL AND coalesce(scope IN ('next_run', 'ongoing'), FALSE)
    END
  ),
  CONSTRAINT decisions_turn_check CHECK (proposal_turn_id != turn_id AND supersedes_id != id)
);

CREATE INDEX ON public.decisions (project_id, decided_at DESC, id DESC);
CREATE INDEX ON public.decisions (proposal_turn_id);

CREATE TABLE public.run_preferences (
  run_id uuid NOT NULL,
  decision_id bigint NOT NULL,
  project_id uuid NOT NULL,
  scope text NOT NULL,
  PRIMARY KEY (run_id, decision_id),
  FOREIGN KEY (run_id, project_id) REFERENCES public.runs (id, project_id) ON DELETE CASCADE,
  FOREIGN KEY (decision_id, project_id, scope) REFERENCES public.decisions (id, project_id, scope)
);

CREATE UNIQUE INDEX run_preferences_one_next_run ON public.run_preferences (decision_id)
WHERE scope = 'next_run';
CREATE INDEX ON public.run_preferences (decision_id);

CREATE TABLE public.assay_results (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  project_id uuid NOT NULL REFERENCES public.projects (id) ON DELETE CASCADE,
  molecule_id bigint NOT NULL REFERENCES public.molecules (id),
  assay text NOT NULL CHECK (
    assay = btrim(assay, E' \t\n\r\f\013') AND length(assay) BETWEEN 1 AND 128
  ),
  endpoint text NOT NULL CHECK (
    endpoint = btrim(endpoint, E' \t\n\r\f\013') AND length(endpoint) BETWEEN 1 AND 80
  ),
  relation text NOT NULL CHECK (relation IN ('=', '<', '<=', '>', '>=')),
  value double precision NOT NULL CHECK (value > '-Infinity' AND value < 'Infinity'),
  unit text NOT NULL CHECK (
    unit = btrim(unit, E' \t\n\r\f\013') AND length(unit) BETWEEN 1 AND 40
  ),
  conditions jsonb NOT NULL CHECK (
    jsonb_typeof(conditions) = 'object' AND octet_length(conditions::text) <= 1024
  ),
  source_ref text NOT NULL CHECK (
    source_ref = btrim(source_ref, E' \t\n\r\f\013') AND length(source_ref) BETWEEN 1 AND 256
  ),
  decision_id bigint REFERENCES public.decisions (id),
  recorded_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (project_id, source_ref)
);

CREATE INDEX ON public.assay_results (project_id, molecule_id, id DESC);
CREATE INDEX ON public.assay_results (molecule_id);
CREATE INDEX ON public.assay_results (decision_id);

CREATE VIEW public.project_molecules WITH (security_invoker = TRUE) AS
SELECT
  projects.id AS project_id,
  projects.source_molecule_id AS molecule_id
FROM public.projects
UNION
SELECT
  runs.project_id,
  candidates.molecule_id
FROM public.runs
INNER JOIN public.candidates
  ON candidates.run_id = runs.id
WHERE runs.project_id IS NOT NULL;

CREATE VIEW public.current_decisions WITH (security_invoker = TRUE) AS
SELECT
  decisions.id,
  decisions.turn_id,
  decisions.project_id,
  decisions.decision_type,
  decisions.proposal_turn_id,
  decisions.supersedes_id,
  decisions.note,
  decisions.scope,
  decisions.decided_at
FROM public.decisions
WHERE NOT EXISTS (
  SELECT 1
  FROM public.decisions AS newer
  WHERE newer.supersedes_id = decisions.id
);

CREATE VIEW public.session_turns WITH (security_invoker = TRUE) AS
SELECT
  turns.id,
  turns.session_id,
  turns.project_id,
  turns.turn_index,
  turns.request_id,
  turns.request,
  turns.status,
  turns.task,
  turns.output,
  turns.trace,
  turns.started_at,
  turns.finished_at,
  runs.id AS run_id,
  decisions.id AS decision_id,
  ARRAY(
    SELECT proposals.molecule_id
    FROM public.proposals
    WHERE proposals.turn_id = turns.id
    ORDER BY proposals.ordinal
  ) AS comparison_ids
FROM public.turns
LEFT JOIN public.runs
  ON runs.turn_id = turns.id
LEFT JOIN public.decisions
  ON decisions.turn_id = turns.id;

ALTER TABLE public.molecules ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.molecule_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.turns ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.candidates ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.model_calls ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.proposals ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.decisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.run_preferences ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.assay_results ENABLE ROW LEVEL SECURITY;

CREATE POLICY "read own molecules" ON public.molecules FOR SELECT TO authenticated
USING (user_id = (SELECT auth.uid()));

CREATE POLICY "read own molecule profiles" ON public.molecule_profiles FOR SELECT TO authenticated
USING (molecule_id IN (SELECT id FROM public.molecules WHERE user_id = (SELECT auth.uid())));

CREATE POLICY "read own projects" ON public.projects FOR SELECT TO authenticated
USING (user_id = (SELECT auth.uid()));

CREATE POLICY "read own sessions" ON public.sessions FOR SELECT TO authenticated
USING (project_id IN (SELECT id FROM public.projects WHERE user_id = (SELECT auth.uid())));

CREATE POLICY "read own turns" ON public.turns FOR SELECT TO authenticated
USING (project_id IN (SELECT id FROM public.projects WHERE user_id = (SELECT auth.uid())));

CREATE POLICY "read own runs" ON public.runs FOR SELECT TO authenticated
USING (user_id = (SELECT auth.uid()));

CREATE POLICY "read own candidates" ON public.candidates FOR SELECT TO authenticated
USING (run_id IN (SELECT id FROM public.runs WHERE user_id = (SELECT auth.uid())));

CREATE POLICY "read own model calls" ON public.model_calls FOR SELECT TO authenticated
USING (
  turn_id IN (
    SELECT id FROM public.turns
    WHERE project_id IN (SELECT id FROM public.projects WHERE user_id = (SELECT auth.uid()))
  )
  OR run_id IN (SELECT id FROM public.runs WHERE user_id = (SELECT auth.uid()))
);

CREATE POLICY "read own proposals" ON public.proposals FOR SELECT TO authenticated
USING (
  turn_id IN (
    SELECT id FROM public.turns
    WHERE project_id IN (SELECT id FROM public.projects WHERE user_id = (SELECT auth.uid()))
  )
);

CREATE POLICY "read own decisions" ON public.decisions FOR SELECT TO authenticated
USING (project_id IN (SELECT id FROM public.projects WHERE user_id = (SELECT auth.uid())));

CREATE POLICY "read own run preferences" ON public.run_preferences FOR SELECT TO authenticated
USING (project_id IN (SELECT id FROM public.projects WHERE user_id = (SELECT auth.uid())));

CREATE POLICY "read own assay results" ON public.assay_results FOR SELECT TO authenticated
USING (project_id IN (SELECT id FROM public.projects WHERE user_id = (SELECT auth.uid())));

CREATE FUNCTION public.save_molecule(
  p_user_id uuid,
  p_canonical_smiles text,
  p_standardization_version text,
  p_profile jsonb
)
RETURNS bigint
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
  v_molecule_id bigint;
BEGIN
  INSERT INTO public.molecules (user_id, canonical_smiles, standardization_version)
  VALUES (p_user_id, p_canonical_smiles, p_standardization_version)
  ON CONFLICT (user_id, canonical_smiles) DO UPDATE
    SET standardization_version = excluded.standardization_version
  RETURNING id INTO v_molecule_id;

  INSERT INTO public.molecule_profiles (
    molecule_id, molecular_weight_g_mol, logp, tpsa_angstrom_sq, heavy_atom_count,
    h_bond_acceptors, h_bond_donors, rotatable_bonds, ring_count, aromatic_rings,
    qed, lipinski_violations, sa_score, pains_alerts, predictions
  )
  SELECT
    v_molecule_id, molecular_weight_g_mol, logp, tpsa_angstrom_sq, heavy_atom_count,
    h_bond_acceptors, h_bond_donors, rotatable_bonds, ring_count, aromatic_rings,
    qed, lipinski_violations, sa_score, pains_alerts, predictions
  FROM jsonb_populate_record(NULL::public.molecule_profiles, p_profile)
  ON CONFLICT (molecule_id) DO UPDATE SET
    molecular_weight_g_mol = excluded.molecular_weight_g_mol,
    logp = excluded.logp,
    tpsa_angstrom_sq = excluded.tpsa_angstrom_sq,
    heavy_atom_count = excluded.heavy_atom_count,
    h_bond_acceptors = excluded.h_bond_acceptors,
    h_bond_donors = excluded.h_bond_donors,
    rotatable_bonds = excluded.rotatable_bonds,
    ring_count = excluded.ring_count,
    aromatic_rings = excluded.aromatic_rings,
    qed = excluded.qed,
    lipinski_violations = excluded.lipinski_violations,
    sa_score = excluded.sa_score,
    pains_alerts = excluded.pains_alerts,
    predictions = molecule_profiles.predictions || (
      SELECT coalesce(jsonb_object_agg(item.key, item.value), '{}'::jsonb)
      FROM jsonb_each(excluded.predictions) AS item
      WHERE item.value ->> 'status' = 'computed'
        OR NOT molecule_profiles.predictions ? item.key
    ),
    computed_at = excluded.computed_at;

  RETURN v_molecule_id;
END;
$$;

CREATE FUNCTION public.save_candidates(p_run_id uuid, p_user_id uuid, p_candidates jsonb)
RETURNS void
LANGUAGE plpgsql
SET search_path = ''
AS $$
DECLARE
  v_candidate jsonb;
BEGIN
  FOR v_candidate IN
    SELECT value FROM jsonb_array_elements(p_candidates) ORDER BY value ->> 'canonical_smiles'
  LOOP
    INSERT INTO public.candidates (run_id, molecule_id, ordinal, design, predictions)
    VALUES (
      p_run_id,
      public.save_molecule(
        p_user_id, v_candidate ->> 'canonical_smiles',
        v_candidate ->> 'standardization_version', v_candidate -> 'profile'
      ),
      (v_candidate ->> 'ordinal')::integer,
      nullif(v_candidate -> 'design', 'null'::jsonb),
      nullif(v_candidate -> 'predictions', 'null'::jsonb)
    );
  END LOOP;
END;
$$;

CREATE FUNCTION public.save_model_calls(p_turn_id bigint, p_run_id uuid, p_calls jsonb)
RETURNS void
LANGUAGE sql
SET search_path = ''
AS $$
  INSERT INTO public.model_calls (
    turn_id, run_id, call_index, agent_role, model, status,
    invocation_digest, context_bytes, elapsed_ms, usage, output,
    request_json, validation_status, validation_issues
  )
  SELECT
    p_turn_id, p_run_id, call_index, agent_role, model, status,
    invocation_digest, context_bytes, elapsed_ms, usage, output,
    request_json, validation_status, validation_issues
  FROM jsonb_to_recordset(p_calls) AS calls (
    call_index integer, agent_role text, model text, status text, invocation_digest text,
    context_bytes integer, elapsed_ms double precision, usage jsonb, output jsonb,
    request_json text, validation_status text, validation_issues jsonb
  );
$$;

CREATE FUNCTION public.create_project(p_user_id uuid, p_source_molecule_id bigint)
RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
  v_project public.projects%rowtype;
BEGIN
  INSERT INTO public.projects (user_id, source_molecule_id)
  SELECT p_user_id, id FROM public.molecules WHERE id = p_source_molecule_id AND user_id = p_user_id
  RETURNING * INTO v_project;
  IF NOT found THEN
    RAISE SQLSTATE 'PT404' USING MESSAGE = 'The source molecule does not exist.';
  END IF;
  RETURN to_jsonb(v_project);
EXCEPTION WHEN unique_violation THEN
  RAISE SQLSTATE 'PT409' USING MESSAGE = 'The source molecule already has a project.';
END;
$$;

CREATE FUNCTION public.create_session(p_user_id uuid, p_project_id uuid)
RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
  v_session public.sessions%rowtype;
BEGIN
  INSERT INTO public.sessions (project_id, state)
  SELECT id, jsonb_build_object('parent', jsonb_build_object('molecule_id', source_molecule_id))
  FROM public.projects
  WHERE id = p_project_id AND user_id = p_user_id
  RETURNING * INTO v_session;
  IF NOT found THEN
    RAISE SQLSTATE 'PT404' USING MESSAGE = 'The project does not exist.';
  END IF;
  RETURN to_jsonb(v_session);
END;
$$;

CREATE FUNCTION public.expire_turns(p_user_id uuid, p_session_id uuid)
RETURNS void
LANGUAGE sql
SECURITY DEFINER
SET search_path = ''
AS $$
  UPDATE public.turns
  SET
    status = 'failed',
    finished_at = now(),
    output = '{"error": "The turn expired.", "error_code": "timeout"}',
    trace = '{}'
  FROM public.sessions
  INNER JOIN public.projects
    ON projects.id = sessions.project_id
  WHERE sessions.id = p_session_id
    AND projects.user_id = p_user_id
    AND turns.project_id = projects.id
    AND turns.status = 'running'
    AND turns.started_at <= now() - interval '5 minutes';
$$;

CREATE FUNCTION public.expire_runs(p_user_id uuid, p_run_id uuid)
RETURNS boolean
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
BEGIN
  UPDATE public.runs
  SET status = 'failed', finished_at = now(), failure_reason = 'The generation run expired.'
  WHERE user_id = p_user_id
    AND (p_run_id IS NULL OR id = p_run_id)
    AND project_id IS NULL
    AND status = 'running'
    AND created_at <= now() - interval '5 minutes';
  RETURN found;
END;
$$;

CREATE FUNCTION public.start_project_turn(
  p_user_id uuid,
  p_session_id uuid,
  p_request jsonb,
  p_request_id uuid
)
RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
  v_project public.projects%rowtype;
  v_turn public.turns%rowtype;
BEGIN
  SELECT projects.* INTO v_project
  FROM public.projects
  INNER JOIN public.sessions
    ON sessions.project_id = projects.id
  WHERE sessions.id = p_session_id AND projects.user_id = p_user_id
  FOR UPDATE OF projects;
  IF NOT found THEN
    RAISE SQLSTATE 'PT404' USING MESSAGE = 'The session does not exist.';
  END IF;

  PERFORM public.expire_turns(p_user_id, p_session_id);

  IF p_request_id IS NOT NULL THEN
    SELECT * INTO v_turn
    FROM public.turns
    WHERE session_id = p_session_id AND request_id = p_request_id;
    IF found THEN
      IF v_turn.request IS DISTINCT FROM p_request THEN
        RAISE SQLSTATE 'PT409' USING MESSAGE = 'The request ID names another turn.';
      END IF;
      RETURN jsonb_build_object(
        'turn', (SELECT to_jsonb(session_turns) FROM public.session_turns WHERE id = v_turn.id),
        'replay', TRUE
      );
    END IF;
  END IF;

  INSERT INTO public.turns (session_id, project_id, turn_index, request, request_id, status)
  SELECT
    p_session_id, v_project.id, coalesce(max(turn_index), 0) + 1,
    p_request, p_request_id, 'running'
  FROM public.turns
  WHERE session_id = p_session_id
  ON CONFLICT (project_id) WHERE status = 'running' DO NOTHING
  RETURNING * INTO v_turn;
  IF NOT found THEN
    RAISE SQLSTATE 'PT409' USING MESSAGE = 'A turn is already running in this project.';
  END IF;
  RETURN jsonb_build_object(
    'turn', to_jsonb(v_turn),
    'project', to_jsonb(v_project),
    'state', (SELECT state FROM public.sessions WHERE id = p_session_id),
    'replay', FALSE
  );
END;
$$;

CREATE FUNCTION public.save_proposal(p_project_id uuid, p_turn_id bigint, p_comparison_ids bigint[])
RETURNS void
LANGUAGE plpgsql
SET search_path = ''
AS $$
BEGIN
  IF cardinality(p_comparison_ids) < 2 OR EXISTS (
    SELECT 1
    FROM unnest(p_comparison_ids) AS item (molecule_id)
    WHERE NOT EXISTS (
      SELECT 1
      FROM public.project_molecules
      WHERE project_molecules.project_id = p_project_id
        AND project_molecules.molecule_id = item.molecule_id
    )
  ) THEN
    RAISE SQLSTATE 'PT500'
      USING MESSAGE = 'A proposal needs two or more molecules from its project.';
  END IF;
  INSERT INTO public.proposals (turn_id, molecule_id, ordinal)
  SELECT p_turn_id, item.molecule_id, item.ordinal - 1
  FROM unnest(p_comparison_ids) WITH ORDINALITY AS item (molecule_id, ordinal);
END;
$$;

CREATE FUNCTION public.save_turn_run(
  p_user_id uuid,
  p_project public.projects,
  p_turn_id bigint,
  p_generation jsonb,
  p_preference_ids bigint[]
)
RETURNS void
LANGUAGE plpgsql
SET search_path = ''
AS $$
DECLARE
  v_run_id uuid;
  v_source_id bigint := (p_generation ->> 'source_molecule_id')::bigint;
  v_parent_run_id uuid := (p_generation ->> 'parent_run_id')::uuid;
  v_candidates jsonb := p_generation -> 'candidates';
  v_has_candidates boolean;
  v_preferences integer := coalesce(cardinality(p_preference_ids), 0);
  v_applied integer;
BEGIN
  IF jsonb_typeof(v_candidates) IS DISTINCT FROM 'array' THEN
    RAISE SQLSTATE 'PT500' USING MESSAGE = 'The generation payload needs a candidate array.';
  END IF;
  v_has_candidates := jsonb_array_length(v_candidates) > 0;
  IF v_parent_run_id IS NULL AND v_source_id IS DISTINCT FROM p_project.source_molecule_id THEN
    RAISE SQLSTATE 'PT500'
      USING MESSAGE = 'The run source must be the lead or a candidate of the parent run.';
  END IF;
  IF NOT v_has_candidates AND v_preferences > 0 THEN
    RAISE SQLSTATE 'PT500' USING MESSAGE = 'A run without candidates applies no preferences.';
  END IF;

  INSERT INTO public.runs (
    user_id, source_molecule_id, project_id, parent_run_id, turn_id, status, model, config,
    finished_at, input_tokens, output_tokens, reasoning_tokens, cached_tokens
  ) VALUES (
    p_user_id, v_source_id, p_project.id, v_parent_run_id, p_turn_id,
    CASE WHEN v_has_candidates THEN 'completed' ELSE 'limit_exhausted' END,
    p_generation ->> 'model', p_generation -> 'config', now(),
    coalesce((p_generation #>> '{usage,input_tokens}')::integer, 0),
    coalesce((p_generation #>> '{usage,output_tokens}')::integer, 0),
    coalesce((p_generation #>> '{usage,reasoning_tokens}')::integer, 0),
    coalesce((p_generation #>> '{usage,cached_tokens}')::integer, 0)
  )
  RETURNING id INTO v_run_id;

  BEGIN
    INSERT INTO public.run_preferences (run_id, decision_id, project_id, scope)
    SELECT v_run_id, id, project_id, scope
    FROM public.current_decisions
    WHERE project_id = p_project.id AND id = ANY (p_preference_ids) AND scope IS NOT NULL;
    GET DIAGNOSTICS v_applied = ROW_COUNT;
  EXCEPTION WHEN unique_violation THEN
    RAISE SQLSTATE 'PT409' USING MESSAGE = 'A preference no longer applies.';
  END;
  IF v_applied != v_preferences THEN
    RAISE SQLSTATE 'PT409' USING MESSAGE = 'A preference no longer applies.';
  END IF;

  PERFORM public.save_candidates(v_run_id, p_user_id, v_candidates);
END;
$$;

CREATE FUNCTION public.save_decision(p_project_id uuid, p_turn_id bigint, p_action jsonb)
RETURNS void
LANGUAGE plpgsql
SET search_path = ''
AS $$
DECLARE
  v_type text := p_action ->> 'type';
  v_proposal_turn_id bigint := (p_action ->> 'proposal_turn_id')::bigint;
  v_supersedes_id bigint := (p_action ->> 'supersedes_id')::bigint;
BEGIN
  IF v_type = 'accept_proposal' THEN
    IF NOT EXISTS (
      SELECT 1
      FROM public.proposals
      INNER JOIN public.turns
        ON turns.id = proposals.turn_id
      WHERE proposals.turn_id = v_proposal_turn_id AND turns.project_id = p_project_id
    ) THEN
      RAISE SQLSTATE 'PT404' USING MESSAGE = 'The proposal does not exist in this project.';
    END IF;
    IF EXISTS (
      SELECT 1
      FROM public.current_decisions
      WHERE proposal_turn_id = v_proposal_turn_id AND id IS DISTINCT FROM v_supersedes_id
    ) THEN
      RAISE SQLSTATE 'PT409' USING MESSAGE = 'The proposal already has an acceptance.';
    END IF;
    IF v_supersedes_id IS NOT NULL AND NOT EXISTS (
      SELECT 1
      FROM public.decisions
      WHERE id = v_supersedes_id AND proposal_turn_id = v_proposal_turn_id
    ) THEN
      RAISE SQLSTATE 'PT409'
        USING MESSAGE = 'A decision can only replace one for the same proposal.';
    END IF;
  END IF;

  INSERT INTO public.decisions (
    turn_id, project_id, decision_type, proposal_turn_id, supersedes_id, note, scope
  ) VALUES (
    p_turn_id, p_project_id, v_type, v_proposal_turn_id, v_supersedes_id,
    CASE v_type WHEN 'accept_proposal' THEN p_action ->> 'reason' ELSE p_action ->> 'text' END,
    p_action ->> 'scope'
  );
EXCEPTION WHEN foreign_key_violation OR unique_violation THEN
  RAISE SQLSTATE 'PT409'
    USING MESSAGE = 'Choose a current decision of the same type in this project.';
END;
$$;

CREATE FUNCTION public.finish_project_turn(
  p_user_id uuid,
  p_turn_id bigint,
  p_status text,
  p_task text,
  p_output jsonb,
  p_trace jsonb,
  p_state jsonb,
  p_calls jsonb,
  p_comparison_ids bigint[],
  p_preference_ids bigint[],
  p_generation jsonb
)
RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
  v_project public.projects%rowtype;
  v_turn public.turns%rowtype;
  v_decision_task text;
  v_decides boolean;
  v_comparisons integer := coalesce(cardinality(p_comparison_ids), 0);
  v_preferences integer := coalesce(cardinality(p_preference_ids), 0);
  v_has_proposal boolean := jsonb_typeof(p_output -> 'proposal') IS NOT DISTINCT FROM 'object';
BEGIN
  SELECT projects.* INTO v_project
  FROM public.projects
  INNER JOIN public.turns
    ON turns.project_id = projects.id
  WHERE turns.id = p_turn_id AND projects.user_id = p_user_id
  FOR UPDATE OF projects;
  IF NOT found THEN
    RAISE SQLSTATE 'PT404' USING MESSAGE = 'The turn does not exist.';
  END IF;
  SELECT * INTO v_turn FROM public.turns WHERE id = p_turn_id AND status = 'running' FOR UPDATE;
  IF NOT found THEN
    RAISE SQLSTATE 'PT409' USING MESSAGE = 'The turn is no longer running.';
  END IF;

  v_decision_task := CASE v_turn.request #>> '{action,type}'
    WHEN 'accept_proposal' THEN 'decide'
    WHEN 'set_preference' THEN 'remember'
  END;
  v_decides := p_status IS NOT DISTINCT FROM 'completed' AND v_decision_task IS NOT NULL;
  IF (p_task IS NOT DISTINCT FROM 'design') != (p_generation IS NOT NULL)
    OR (v_preferences > 0 AND p_generation IS NULL)
    OR (v_comparisons > 0) != v_has_proposal
    OR (v_comparisons > 0 AND p_task IS DISTINCT FROM 'strategy')
    OR (v_decides AND p_task IS DISTINCT FROM v_decision_task)
    OR (NOT v_decides AND p_task IN ('decide', 'remember'))
  THEN
    RAISE SQLSTATE 'PT500' USING MESSAGE = 'The turn outcome does not match its request.';
  END IF;

  IF v_comparisons > 0 THEN
    PERFORM public.save_proposal(v_project.id, p_turn_id, p_comparison_ids);
  END IF;
  IF p_generation IS NOT NULL THEN
    PERFORM public.save_turn_run(p_user_id, v_project, p_turn_id, p_generation, p_preference_ids);
  END IF;
  IF v_decides THEN
    PERFORM public.save_decision(v_project.id, p_turn_id, v_turn.request -> 'action');
  END IF;
  PERFORM public.save_model_calls(p_turn_id, NULL, p_calls);

  IF p_status = 'completed' THEN
    UPDATE public.sessions SET state = p_state WHERE id = v_turn.session_id;
  END IF;
  UPDATE public.turns
  SET
    status = p_status,
    task = p_task,
    output = (p_output - ARRAY['task', 'generation_run_id', 'decision_id'])
      #- '{proposal,comparison_ids}',
    trace = p_trace,
    finished_at = now()
  WHERE id = p_turn_id;
  RETURN (SELECT to_jsonb(session_turns) FROM public.session_turns WHERE id = p_turn_id);
END;
$$;

CREATE FUNCTION public.start_run(
  p_user_id uuid,
  p_source_molecule_id bigint,
  p_model text,
  p_config jsonb,
  p_request_id uuid,
  p_request jsonb
)
RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
  v_run public.runs%rowtype;
BEGIN
  PERFORM public.expire_runs(p_user_id, NULL);

  INSERT INTO public.runs (user_id, source_molecule_id, status, model, config, request_id, request)
  SELECT p_user_id, id, 'running', p_model, p_config, p_request_id, p_request
  FROM public.molecules
  WHERE id = p_source_molecule_id AND user_id = p_user_id
  ON CONFLICT (user_id, request_id) WHERE request_id IS NOT NULL DO NOTHING
  RETURNING * INTO v_run;
  IF found THEN
    RETURN jsonb_build_object('id', v_run.id, 'replay', FALSE);
  END IF;

  SELECT * INTO v_run FROM public.runs WHERE user_id = p_user_id AND request_id = p_request_id;
  IF NOT found THEN
    RAISE SQLSTATE 'PT404' USING MESSAGE = 'The source molecule does not exist.';
  END IF;
  IF v_run.source_molecule_id IS DISTINCT FROM p_source_molecule_id
    OR v_run.request IS DISTINCT FROM p_request
  THEN
    RAISE SQLSTATE 'PT409' USING MESSAGE = 'The request ID names another run.';
  END IF;
  RETURN jsonb_build_object('id', v_run.id, 'replay', TRUE);
END;
$$;

CREATE FUNCTION public.finish_run(
  p_user_id uuid,
  p_run_id uuid,
  p_status text,
  p_candidates jsonb,
  p_usage jsonb,
  p_failure_reason text,
  p_source_predictions jsonb,
  p_calls jsonb DEFAULT '[]'
)
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
BEGIN
  IF p_status IS NULL
    OR p_status NOT IN ('completed', 'failed', 'limit_exhausted')
    OR jsonb_typeof(p_candidates) IS DISTINCT FROM 'array'
    OR jsonb_typeof(p_calls) IS DISTINCT FROM 'array'
    OR (p_status = 'failed') != (p_failure_reason IS NOT NULL)
    OR (p_source_predictions IS NOT NULL AND jsonb_typeof(p_source_predictions) != 'object')
  THEN
    RAISE SQLSTATE 'PT500' USING MESSAGE = 'Invalid run outcome.';
  END IF;
  IF p_status != 'completed' AND jsonb_array_length(p_candidates) > 0
  THEN
    RAISE SQLSTATE 'PT500' USING MESSAGE = 'The run outcome does not match its candidates.';
  END IF;
  IF public.expire_runs(p_user_id, p_run_id) THEN
    RETURN;
  END IF;

  UPDATE public.runs
  SET
    status = p_status,
    finished_at = now(),
    input_tokens = coalesce((p_usage ->> 'input_tokens')::integer, 0),
    output_tokens = coalesce((p_usage ->> 'output_tokens')::integer, 0),
    reasoning_tokens = coalesce((p_usage ->> 'reasoning_tokens')::integer, 0),
    cached_tokens = coalesce((p_usage ->> 'cached_tokens')::integer, 0),
    failure_reason = p_failure_reason,
    config = CASE
      WHEN p_source_predictions IS NULL THEN config
      ELSE jsonb_set(config, '{source_predictions}', p_source_predictions)
    END
  WHERE id = p_run_id AND user_id = p_user_id AND project_id IS NULL AND status = 'running';
  IF NOT found THEN
    IF p_status = 'failed' THEN
      RETURN;
    END IF;
    RAISE SQLSTATE 'PT409' USING MESSAGE = 'The run is no longer running.';
  END IF;
  PERFORM public.save_candidates(p_run_id, p_user_id, p_candidates);
  PERFORM public.save_model_calls(NULL, p_run_id, p_calls);
END;
$$;

CREATE FUNCTION public.record_assay_result(p_user_id uuid, p_project_id uuid, p_result jsonb)
RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
  v_input public.assay_results%rowtype;
  v_saved public.assay_results%rowtype;
BEGIN
  IF NOT EXISTS (SELECT 1 FROM public.projects WHERE id = p_project_id AND user_id = p_user_id) THEN
    RAISE SQLSTATE 'PT404' USING MESSAGE = 'The project does not exist.';
  END IF;
  v_input := jsonb_populate_record(NULL::public.assay_results, p_result);

  IF NOT EXISTS (
    SELECT 1 FROM public.project_molecules
    WHERE project_id = p_project_id AND molecule_id = v_input.molecule_id
  ) THEN
    RAISE SQLSTATE 'PT404' USING MESSAGE = 'The molecule is outside this project.';
  END IF;
  IF v_input.decision_id IS NOT NULL AND NOT EXISTS (
    SELECT 1
    FROM public.decisions
    INNER JOIN public.proposals
      ON proposals.turn_id = decisions.proposal_turn_id
    WHERE decisions.id = v_input.decision_id
      AND decisions.project_id = p_project_id
      AND proposals.molecule_id = v_input.molecule_id
  ) THEN
    RAISE SQLSTATE 'PT404'
      USING MESSAGE = 'The accepted experiment does not include this molecule in the project.';
  END IF;

  INSERT INTO public.assay_results (
    project_id, molecule_id, assay, endpoint, relation, value, unit, conditions,
    source_ref, decision_id
  ) VALUES (
    p_project_id, v_input.molecule_id, v_input.assay, v_input.endpoint, v_input.relation,
    v_input.value, v_input.unit, v_input.conditions, v_input.source_ref, v_input.decision_id
  )
  ON CONFLICT (project_id, source_ref) DO NOTHING
  RETURNING * INTO v_saved;

  IF NOT found THEN
    SELECT * INTO STRICT v_saved
    FROM public.assay_results
    WHERE project_id = p_project_id AND source_ref = v_input.source_ref;
    IF ROW(
      v_saved.molecule_id, v_saved.assay, v_saved.endpoint, v_saved.relation,
      v_saved.value, v_saved.unit, v_saved.conditions, v_saved.decision_id
    ) IS DISTINCT FROM ROW(
      v_input.molecule_id, v_input.assay, v_input.endpoint, v_input.relation,
      v_input.value, v_input.unit, v_input.conditions, v_input.decision_id
    ) THEN
      RAISE SQLSTATE 'PT409'
        USING MESSAGE = 'This source reference already has a different assay result.';
    END IF;
  END IF;
  RETURN to_jsonb(v_saved);
END;
$$;

CREATE FUNCTION public.project_decisions(
  p_user_id uuid,
  p_project_id uuid,
  p_ids bigint[] DEFAULT NULL,
  p_as_of timestamptz DEFAULT NULL,
  p_limit integer DEFAULT NULL
)
RETURNS jsonb
LANGUAGE sql
STABLE
SECURITY INVOKER
SET search_path = ''
AS $$
  WITH history AS (
    SELECT decisions.*
    FROM public.decisions
    INNER JOIN public.projects
      ON projects.id = decisions.project_id
    WHERE decisions.project_id = p_project_id
      AND projects.user_id = p_user_id
      AND decisions.decided_at <= coalesce(p_as_of, now())
  ),
  selected AS (
    SELECT history.*
    FROM history
    WHERE CASE
      WHEN cardinality(p_ids) > 0 THEN history.id = ANY (p_ids)
      ELSE NOT EXISTS (SELECT 1 FROM history AS newer WHERE newer.supersedes_id = history.id)
    END
    ORDER BY history.decided_at DESC, history.id DESC
    LIMIT least(greatest(coalesce(p_limit, 100), 1), 100)
  )
  SELECT coalesce(
    jsonb_agg(
      to_jsonb(selected) || jsonb_build_object(
        'proposal', (session_turns.output -> 'proposal')
          || jsonb_build_object('comparison_ids', session_turns.comparison_ids),
        'applied', EXISTS (
          SELECT 1
          FROM public.run_preferences
          INNER JOIN public.runs
            ON runs.id = run_preferences.run_id
          WHERE run_preferences.decision_id = selected.id
            AND runs.finished_at <= coalesce(p_as_of, now())
        )
      )
      ORDER BY selected.decided_at DESC, selected.id DESC
    ),
    '[]'::jsonb
  )
  FROM selected
  LEFT JOIN public.session_turns
    ON session_turns.id = selected.proposal_turn_id;
$$;

REVOKE ALL ON TABLE
  public.molecules, public.molecule_profiles, public.projects, public.sessions,
  public.turns, public.runs, public.candidates, public.model_calls, public.proposals,
  public.decisions, public.run_preferences, public.assay_results, public.project_molecules,
  public.current_decisions, public.session_turns
FROM PUBLIC, anon, authenticated, service_role;

REVOKE ALL ON SEQUENCE
  public.molecules_id_seq, public.turns_id_seq, public.model_calls_id_seq,
  public.decisions_id_seq, public.assay_results_id_seq
FROM PUBLIC, anon, authenticated, service_role;

REVOKE ALL ON FUNCTION
  public.save_molecule (uuid, text, text, jsonb),
  public.save_candidates (uuid, uuid, jsonb),
  public.save_model_calls (bigint, uuid, jsonb),
  public.create_project (uuid, bigint),
  public.create_session (uuid, uuid),
  public.expire_turns (uuid, uuid),
  public.expire_runs (uuid, uuid),
  public.start_project_turn (uuid, uuid, jsonb, uuid),
  public.save_proposal (uuid, bigint, bigint[]),
  public.save_turn_run (uuid, public.projects, bigint, jsonb, bigint[]),
  public.save_decision (uuid, bigint, jsonb),
  public.finish_project_turn (
    uuid, bigint, text, text, jsonb, jsonb, jsonb, jsonb, bigint[], bigint[], jsonb
  ),
  public.start_run (uuid, bigint, text, jsonb, uuid, jsonb),
  public.finish_run (uuid, uuid, text, jsonb, jsonb, text, jsonb, jsonb),
  public.record_assay_result (uuid, uuid, jsonb),
  public.project_decisions (uuid, uuid, bigint[], timestamptz, integer)
FROM PUBLIC, anon, authenticated, service_role;

GRANT SELECT ON TABLE
  public.molecules, public.molecule_profiles, public.projects, public.sessions,
  public.turns, public.runs, public.candidates, public.model_calls, public.proposals,
  public.decisions, public.run_preferences, public.assay_results, public.project_molecules,
  public.current_decisions, public.session_turns
TO authenticated, service_role;

GRANT EXECUTE ON FUNCTION
  public.save_molecule (uuid, text, text, jsonb),
  public.create_project (uuid, bigint),
  public.create_session (uuid, uuid),
  public.expire_turns (uuid, uuid),
  public.start_project_turn (uuid, uuid, jsonb, uuid),
  public.finish_project_turn (
    uuid, bigint, text, text, jsonb, jsonb, jsonb, jsonb, bigint[], bigint[], jsonb
  ),
  public.expire_runs (uuid, uuid),
  public.start_run (uuid, bigint, text, jsonb, uuid, jsonb),
  public.finish_run (uuid, uuid, text, jsonb, jsonb, text, jsonb, jsonb),
  public.record_assay_result (uuid, uuid, jsonb)
TO service_role;

GRANT EXECUTE ON FUNCTION public.project_decisions (uuid, uuid, bigint[], timestamptz, integer)
TO authenticated, service_role;
