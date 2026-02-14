ALTER TABLE risk_signal_embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE household_calibration ENABLE ROW LEVEL SECURITY;

CREATE POLICY risk_signal_embeddings_all ON risk_signal_embeddings FOR ALL USING (household_id = public.user_household_id());
CREATE POLICY household_calibration_all ON household_calibration FOR ALL USING (household_id = public.user_household_id());
