# TrashVision_Nairobi



REFINED SOFTWARE DESIGN DOCUMENT
TrashVision Nairobi
AI-Powered Organic Waste and Drainage Intelligence Platform
MVP promise: Within one week, the team will demonstrate image reporting, AI-assisted waste and drainage detection, an explainable severity score, and a Nairobi market hotspot map.

Document type: Refined SDD + 3-person build plan
Prepared for: Michael , Patricia and Levy
Date: 06 July 2026
Team size: 3 people
Core principle: build a narrow vertical slice first, then polish. The demo must work even if the model is imperfect.
 1. Clean Summary
TrashVision Nairobi is a mobile-friendly platform for reporting organic waste, drainage blockage, and standing water in Nairobi markets. A reporter submits a photo and location and date. The system stores the report, runs AI-assisted detection, calculates a 0-100 severity score, and shows the result on a dashboard and map.

1.1 What the Project Is Solving
1.	Market waste and blocked drains are often noticed by people on the ground, but the information is scattered across word of mouth, WhatsApp posts, and delayed reports.
2.	Cleanup teams need a simple way to see where the most serious visible problems are, which locations need review first, and which reports are unclear.
3.	The project turns citizen or trader photos into organized evidence: location, AI suggestion, confidence, severity, status, and map visibility.
1.2 MVP Success in One Week
Success area	Refined target
Report flow	A user submits a JPG/PNG/WebP image, market name, GPS/manual location, and optional note from phone or laptop.
AI result	The backend returns one or more suggested classes: organic_waste, drain_blockage, or standing_water, with confidence and model version.
Severity	The system computes a deterministic 0-100 score and priority level: low, medium, high, or urgent.
Dashboard	A coordinator can filter reports by market, status, or priority and see them on a Leaflet/OpenStreetMap map.
Demo honesty	The team explains limitations clearly and includes at least one low-confidence or failed example.

1.3 Recommended Beginner-Friendly Stack
Layer	Choice	Reason
Frontend	Next.js + React + Tailwind CSS	Responsive web app for report form, dashboard, detail page, and map.
Backend/API	Python FastAPI	Works well with ML code, supports clear schemas, and gives automatic API docs.
AI/ML	Small pretrained YOLO model	Fast object detection demo using transfer learning and visible bounding boxes.
Data/storage	PostgreSQL + object storage, e.g. Supabase	Stores report metadata and images without heavy infrastructure.
Map	Leaflet + OpenStreetMap	Open mapping tools with markers and popups.
Deployment	Vercel/Netlify for web + Render/Railway/Fly.io for API	Simple hosted demo path with health checks and environment variables.

 2. Refined Product Scope
2.1 Goals
4.	Make visible waste and drainage issues easier to report and prioritize.
5.	Use AI as an assistant, not a final authority.
6.	Create an explainable cleanup-priority view for market hotspots.
7.	Build a foundation that can later support county workflows, circular-economy partners, or WhatsApp reporting.
2.2 MVP In Scope
Feature	MVP detail
Report submission	Image upload, market, GPS/manual pin, optional note, privacy warning, upload progress , date+.
AI classes	organic_waste, drain_blockage, standing_water.
Scoring	Severity score, flood-risk proxy, priority label, confidence, and explanation.
Review states	pending, analyzed, needs_review, reviewed, failed.
Dashboard/map	Report list, filters, detail page, map markers, and fallback list if the map fails.
Demo assets	At least five approved demo images, seed data, README, model card, and backup video.

2.3 Out of Scope for Week One
8.	No automatic dispatch to Nairobi County or any cleanup team.
9.	No SMS/WhatsApp bot, payments, rewards, or full citizen accounts.
10.	No scientifically validated weather-based flood forecasting.
11.	No native mobile app or offline synchronization.
12.	No marketplace for waste buyers in the MVP, only a possible recommendation label.
2.4 Primary Users
User	Need	MVP journey
Reporter	Record a visible problem quickly.	Open form -> add photo/location -> submit -> receive result.
Cleanup coordinator	See what needs review first.	Open dashboard -> filter by priority -> inspect map/detail.
Model reviewer	Catch weak AI predictions.	Open low-confidence report -> compare image/result -> mark reviewed.
Judge/stakeholder	Understand value and feasibility.	Watch report-to-map demo -> hear impact, limits, and next steps.

 3. Refined Software Design Document
3.1 System Architecture
Architecture decision: Use a modular monolith for the sprint: one responsive web client, one FastAPI service, one PostgreSQL database, one image bucket, and one ML inference module hidden behind a service interface.

Step	Owner component	What happens
1	Reporter + Web client	Reporter selects a photo, confirms market/location, and submits the report.
2	FastAPI	Validates file and location, stores metadata, uploads image, creates pending report.
3	ML inference module	Loads model, preprocesses image, returns class detections and confidence.
4	Scoring module	Converts detections into severity, flood-risk proxy, priority, and explanation.
5	Database + dashboard	Stores prediction, updates status, and displays list/map/detail views.

3.2 Component Responsibilities
Component	Owns	Does not own
Web client	Report form, preview, browser validation, dashboard, map, detail page, user messages.	Secrets, final scoring, model inference.
FastAPI backend	Routes, validation, report lifecycle, storage/database access, request IDs, errors.	Map rendering or raw model training.
ML module	Model loading, preprocessing, detections, inference timing, model version.	Database writes or UI labels.
Scoring module	Deterministic severity and risk-proxy formula with explanation.	Training data collection.
PostgreSQL	Report metadata, prediction JSON, status, timestamps, model version.	Binary image content.
Object storage	Image objects with generated keys and safe access policy.	Business rules and report status.

3.3 Core Data Model
Entity	Important fields	Notes
reports	id, market, latitude, longitude, note, image_key/url, status, created_at, reviewed_at	Main record created from every submission.
predictions	id, report_id, model_version, classes_json, max_confidence, severity_score, risk_proxy, priority, explanation, inference_ms	Keep detections as JSON during the sprint.
review_events	id, report_id, old_status, new_status, reviewer_alias, created_at	Optional in MVP, useful for audit trail.

Status lifecycle: pending -> analyzed -> reviewed. Alternate paths: pending -> failed, analyzed -> needs_review, and failed -> pending after retry.
3.4 API Contract
Method	Endpoint	Purpose	Success
POST	/api/v1/reports	Create report, upload image, begin analysis.	201 + report
GET	/api/v1/reports	List reports; filter by status, priority, market.	200 + items
GET	/api/v1/reports/{id}	Return report, prediction, and explanation.	200 + report
POST	/api/v1/reports/{id}/retry	Retry a failed analysis.	202 + pending
PATCH	/api/v1/reports/{id}/review	Mark reviewed or needs_review.	200 + report
GET	/health	Process and dependency health.	200 + status

Error rule: Every API error should return code, message, field when relevant, and request_id. Never return stack traces or secrets to the browser.

 4. AI/ML and Scoring Design
4.1 Problem Formulation
Use object detection with three labels: organic_waste, drain_blockage, and standing_water. Start with a small pretrained YOLO model and fine-tune it using a small but carefully labelled dataset. If training blocks progress, use the fallback ladder instead of pretending the model is accurate.
4.2 Minimum Dataset Plan
Task	Minimum target	Quality rule
Collect	150-300 consented or licensed images across all three labels.	Include different lighting, angles, distances, markets, and clean negatives.
Annotate	Bounding boxes with one shared label guide.	Peer-check at least 20 percent of labels.
Split	70 percent train, 20 percent validation, 10 percent test.	Keep near-duplicates and same scenes in the same split to avoid leakage.
Train	Short transfer-learning runs with saved config and seed.	Keep the best validation checkpoint, not automatically the last epoch.
Evaluate	Precision, recall, mAP, confusion review, and 10-20 visual examples.	Show both success and failure examples.

4.3 Explainable Scoring
Severity score: 0.45 x waste coverage + 0.25 x blockage evidence + 0.20 x standing-water evidence + 0.10 x confidence.

Flood-risk proxy: 0.55 x blockage evidence + 0.35 x standing-water evidence + 0.10 x severity. This is a visible-condition proxy, not a weather forecast.

Score	Priority	UI treatment
0-29	Low	Monitor; normal marker.
30-54	Medium	Routine review queue.
55-79	High	Prominent marker; cleanup review recommended.
80-100	Urgent	Immediate human verification; no automatic dispatch in MVP.

4.4 Responsible AI Rules
13.	Show confidence and model version with every prediction.
14.	Use wording like AI suggestion, not AI certainty.
15.	Send ambiguous or low-confidence reports to needs_review.
16.	Do not collect names or phone numbers in the MVP.
17.	Warn users not to photograph faces, vehicle plates, or personal details.
18.	Keep a model card: dataset source, label definitions, split method, training configuration, metrics, failure modes, and intended use.
 5. Frontend, Dashboard, and Testing
5.1 Screens
Screen	Essential elements
Report	Image picker/camera, preview, market selector, GPS/manual pin, note, privacy hint, progress, result.
Dashboard	Summary counts, priority/status/market filters, report cards or table, loading/empty/error states.
Map	Leaflet map, markers by priority, popup with market, score, status, and details link.
Report detail	Image, detection summary, confidence, model version, score explanation, status, review action.

5.2 Testing Strategy
Level	What to test	Exit criterion
Unit	Scoring boundaries, validators, schemas.	Boundary cases pass locally or in CI.
API	Valid create/list/detail, invalid file/location, failed inference.	Core endpoints pass automated tests.
ML smoke	Model loads once and returns documented schema.	Five demo images complete without crashing.
Integration	Upload -> storage -> inference -> database -> dashboard.	Three end-to-end reports persist and render.
UI	Mobile form, filters, map popup, loading/empty/error states.	360px phone-width walkthrough has no blocker.
Demo rehearsal	Fresh deployment, seed data, and backup assets.	Two timed demos finish under five minutes.

5.3 Definition of Done
19.	Works on another teammate's machine or the shared deployment.
20.	Has an acceptance check or test and no secrets committed.
21.	Handles loading, empty, and error states where relevant.
22.	Is reviewed by one teammate before merging.
23.	Is connected to the main demo path, not hidden on a lonely branch.
5.4 Demo Recovery Plan
24.	Keep five approved demo images locally and in a safe demo folder.
25.	Seed the database before presentation with representative reports from Wakulima, Gikomba, and Muthurwa.
26.	If live inference fails, show stored predictions and explain that the inference service failed honestly.
27.	Keep a two-minute backup screen recording, but present live if possible.
 6. 3-Person Team Roles
Team setup: Since the team is you plus two teammates, each person owns one lane but helps with integration daily. The project wins when the lanes meet in one working demo, not when everyone builds separate islands.

Person	Main role	Owns	Helps with
Levis	Product lead + Frontend/Demo	Scope, user stories, report form, dashboard UI, map UI, pitch, rehearsal, GitHub task board.	Testing, seed data, final integration, README clarity.
Patricia	Backend + Database/Deployment	FastAPI, validation, PostgreSQL schema, image storage, API endpoints, logs, deployment, environment variables.	Frontend API wiring and demo recovery.
Michael	AI/ML + Dataset/Scoring	Image collection, annotation guide, YOLO training, inference module, scoring formula, model card, failure examples.	Backend inference contract and dashboard result language.

6.1 Collaboration Rules
28.	One repository, one task board, and one shared README.
29.	Daily 20-minute standup: yesterday, today, blocker, integration risk.
30.	Every day ends with a merged or demoable vertical slice, even if small.
31.	Use branches: Feature/Patricia, Feature/Michael, Feature/Levis.
32.	No secrets in GitHub. Use .env locally and .env.example for names only.
33.	No midnight heroics: if a model or deployment blocks the demo, use the fallback ladder early.
6.2 RACI Snapshot
Workstream	Responsible	Accountable	Consulted	Informed
Product scope and pitch	Levis	Levis	All	All
Frontend and map	Levis	Levis	Patricia	Patrcia
Backend API and database	Patrcia	Patricia	Michael	Michael
ML model and dataset	Michael	Michael	Patricia	Patrcia
Scoring and explanation	Michael + Patrcia	levis	All	All
Deployment and demo backup	Patrcia + Levi	Michael	Michael	All

 7. One-Week Build Plan for 3 People
This plan converts the original four-person routine into a three-person sprint. Keep the goal narrow: upload, detect, score, show on dashboard/map, explain honestly.
Day	Levi: Product + Frontend	Patricia: Backend + Data	Michael: ML + Dataset
Day 1 - Align and scaffold	Freeze MVP, write 3 user stories, create task board, scaffold Next.js/Tailwind pages: report, dashboard, map.	Scaffold FastAPI, /health, settings, database connection stub, .env.example, basic project structure.	Write label guide, collect starter images, test YOLO tooling on one sample, define train/validation/test lesson.
Day 2 - Upload and persistence	Finish report form: image preview, market, GPS/manual location, progress and errors. Connect to API.	Create reports table, storage bucket logic, POST/GET reports endpoints, validation for file and location.	Collect/annotate first balanced batch, peer-check labels, create dataset config and tiny training smoke test.
Day 3 - Baseline prediction	Build result cards using mocked JSON: confidence, priority, needs-review language, model version.	Create model service interface, prediction schema, scoring module shell, failed/pending states.	Train baseline model, evaluate per class, export best checkpoint, prepare 10-20 untouched examples.
Day 4 - Vertical slice	Connect result/detail pages, review action UI, loading/pending/failed states. Test phone width.	Wire inference into report creation, persist prediction, add detail/retry/review endpoints, request IDs and logs.	Tune threshold from validation evidence, add low-confidence behavior, write score explanations for demo cases.
Day 5 - Dashboard and deployment	Finish dashboard counts/filters and Leaflet markers/popups. Deploy web client.	Deploy API, configure database/storage/CORS, add health checks and safe upload limits.	Add model warm-up, record inference time, finish model card and limitations.
Day 6 - Test and story	Lead usability test, polish mobile UI, write 5-minute pitch, rehearse narrator flow.	Run API/integration tests, check logs and failure recovery, freeze backend.	Run final test-set evaluation once, choose honest success/failure examples, freeze model.
Day 7 - Final demo	Drive final rehearsal, open required tabs, silence notifications, deliver story and demo.	Verify deployment, seed database, keep backup data/video ready, monitor API during demo.	Explain AI method, dataset, score formula, limitations, and next steps.

7.1 Daily Exit Checks
Day	Exit check
1	Web and API start from README; /health works; scope frozen.
2	One real image upload persists after refresh; bad files are rejected clearly.
3	Five local images produce structured prediction JSON; score unit tests pass.
4	Three end-to-end reports succeed; one failed or ambiguous case is handled honestly.
5	Public demo URLs work; map markers match records; mobile layout is usable.
6	No blocker remains; two timed rehearsals finish under five minutes.
7	Team can explain what works, what does not, how it was tested, and what comes next.

 8. Fallback Ladder and Risk Register
8.1 Fallback Ladder
If this breaks...	Use this fallback	How to present it honestly
Fine-tuning is too slow	Use pretrained model or simple classifier for demo plus manual labelled examples.	State that custom fine-tuning is planned and show dataset/model-card work.
Model confidence is weak	Keep predictions but mark many reports as needs_review.	Explain that human review is part of the design.
Live inference fails	Show stored predictions seeded from approved demo images.	Say the service failed, but the data model and UI flow still work.
Map tiles fail	Use dashboard fallback list sorted by priority.	Explain map dependency and still show decision support value.
Deployment fails	Run local demo and show backup recording.	Explain deployment path and show working local system.

8.2 Highest Risks
Risk	Impact	Control
Weak dataset	Bad model output and weak judging confidence.	Collect consented images early, peer-check labels, show failure examples.
Overclaiming flood prediction	Credibility and safety risk.	Call it a visible-condition risk proxy only.
Scope creep	Demo breaks before judging.	Freeze MVP after Day 1 and backlog everything else.
Privacy exposure	Faces, plates, or personal details in demo.	Use consented images, warn users, delete demo data after event.
Team integration delays	Separate pieces do not work together.	Integrate every day and test on two machines.

8.3 Final Acceptance Checklist
34.	Upload to dashboard/map works without manual database edits.
35.	At least five approved images are tested, including one failure or low-confidence case.
36.	Severity formula, thresholds, confidence, and model version are visible and documented.
37.	No secrets, personal data, or unlicensed images are committed or shown.
38.	README setup works for a teammate and core tests pass.
39.	The pitch separates MVP decision support from future county integration and flood forecasting.
40.	The backup demo path is ready and the walkthrough finishes under five minutes.
 9. Post-Hackathon Roadmap
Phase	Next improvement
Phase 1: Stabilize	Improve dataset quality, add reviewer accounts, harden upload security, and add monitoring.
Phase 2: Community reporting	Add WhatsApp intake, duplicate detection, language-friendly forms, and reporter feedback.
Phase 3: Operations	Add cleanup assignment, status updates, service-level reports, and partnership dashboards.
Phase 4: Intelligence	Add historical hotspots, rainfall/drainage data only if validated, and better model evaluation by market/context.
Phase 5: Circular economy	Recommend organic waste reuse partners, composting opportunities, and aggregated waste trends.

10. Final Refined Pitch
Pitch: TrashVision Nairobi helps markets turn scattered waste and drainage observations into structured, reviewable evidence. A reporter uploads a photo and location; the system gives an AI suggestion, confidence, severity score, and map marker. The goal is not to replace humans, but to help coordinators see where visible problems need attention first. The MVP proves the report-to-map workflow, the AI-assistance layer, and the responsible review process in one practical demo.

Recommended demo order:
41.	Show the market problem in one sentence.
42.	Submit a new report from a phone-width browser.
43.	Show the AI suggestion, confidence, score, and status.
44.	Open dashboard and map, then filter by high priority.
45.	Show one failure/needs-review case and explain responsible AI.
46.	End with the roadmap: better data, WhatsApp intake, county/circular-economy partnerships.
