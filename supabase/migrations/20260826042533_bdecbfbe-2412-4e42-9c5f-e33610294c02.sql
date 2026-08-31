CREATE OR REPLACE FUNCTION public.record_analysis_run(_kind text, _label text, _status text, _duration_ms integer, _error text, _metadata jsonb)
RETURNS uuid LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
DECLARE _uid uuid := auth.uid();
        _run uuid;
        _q public.usage_quotas%ROWTYPE;
BEGIN
  IF _uid IS NULL THEN RAISE EXCEPTION 'not authenticated'; END IF;

  INSERT INTO public.usage_quotas (user_id) VALUES (_uid) ON CONFLICT (user_id) DO NOTHING;
  SELECT * INTO _q FROM public.usage_quotas WHERE user_id = _uid FOR UPDATE;

  IF _q.period_start < date_trunc('month', now())::date THEN
    UPDATE public.usage_quotas SET used_this_period = 0, period_start = date_trunc('month', now())::date, updated_at = now() WHERE user_id = _uid;
    _q.used_this_period := 0;
  END IF;

  IF _q.used_this_period >= _q.monthly_limit THEN
    RAISE EXCEPTION 'monthly run limit reached';
  END IF;

  INSERT INTO public.analysis_runs (user_id, kind, label, status, duration_ms, error_message, metadata)
  VALUES (_uid, _kind, nullif(btrim(coalesce(_label,'')),''), COALESCE(nullif(_status,''),'started'), nullif(_duration_ms,0), nullif(btrim(coalesce(_error,'')),''), COALESCE(_metadata,'{}'::jsonb))
  RETURNING id INTO _run;

  UPDATE public.usage_quotas SET used_this_period = used_this_period + 1, updated_at = now() WHERE user_id = _uid;
  UPDATE public.profiles SET last_seen_at = now() WHERE id = _uid;

  RETURN _run;
END;
$$;

REVOKE ALL ON FUNCTION public.record_analysis_run(text,text,text,integer,text,jsonb) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.record_analysis_run(text,text,text,integer,text,jsonb) TO authenticated;