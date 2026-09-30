# EXAMPLE — HeartMuLa stays on local heartlib

- **status:** example
- **date:** 2026-09-30
- **tried:** heartlib `HeartMuLaGenPipeline` / `HeartTranscriptorPipeline` versus driving the tower Comfy classes, and `HeartCodec-oss` versus `HeartCodec-oss-20260123`
- **chosen:** Buddy calls heartlib. Comfy classes stay a tower registration (`HeartMuLa_Generate`, `HeartMuLa_Transcribe` only). Codec download id is `HeartMuLa/HeartCodec-oss-20260123` into the `HeartCodec-oss` folder. Default checkpoint is `HeartMuLa-oss-3B-happy-new-year` into `HeartMuLa-oss-3B`.
- **why:** This note is an integration choice from the heartlib README and Hub listings, not a generation that was run. Weights were not fetched. `HeartCodec-oss` returned 401 on the tower and in the Hub lookup, so filenames for that id are not attested. Tower F: had about 28GB free; the default pull is about 25GB, so a space check stays in front of `--yes`.
- **revisit when:** a live generate or transcribe is actually run, or the unsuffixed codec listing becomes readable
