# Coding rules: government crises and ParlGov cabinet changes

Applies to `crises.csv` (149 rows, 23 countries).

## 1. Unit, schema and sources

- One row per ParlGov candidate cabinet change (`event_id` = country + `parlgov_cabinet_id`). No rows are added outside the candidate list.
- ParlGov ids, dates (`parlgov_start_date`) and cabinet names are never changed by coding. Join to ParlGov on `parlgov_cabinet_id`, not on names: some candidate names carry trailing spaces (e.g. "Lofven IV ").
- Cabinets that start before a country's index data begin get a minimal row: `is_crisis` empty, not coded (23 rows). Whether a row enters the estimation window is decided in the analysis code.
- Every date comes from a page opened during coding, never from memory. Pages seen only in search results give no date.
- `evidence` is an exact excerpt of at most 25 words from `source_url`; `second_source_url` holds a second opened page where one was used.

## 2. The is_crisis principle

- **is_crisis = 1**: an unplanned event that changes, or puts at risk, who governs: the head of government, the parties in government, or the parliamentary majority.
- News valence and whether the government's survival was actually threatened do not matter.
- Voluntary exits stay 1. Robustness set R_vol isolates those that came without pressure on the government's position.
- **is_crisis = 0**: planned or scheduled changes (elections at term end, constitutionally required re-appointments, planned successions, formations) and changes where who governs stays the same.
- **is_crisis = ambiguous**: only for event_type OTHER, or as an explicit exception (section 3).

## 3. event_type table

| event_type | Definition | Default is_crisis |
|---|---|---|
| LOST_MAJORITY | A confidence or censure vote is lost, or the majority is lost without a formal partner exit (defections, a split that removes the majority). | 1 |
| COALITION_EXIT | A partner party leaves the government or is expelled, including PM-initiated expulsions, whatever happens to the majority. | 1 |
| LEADERSHIP | The head of government leaves or announces departure outside a schedule: forced, unexpected or voluntary-unexpected (rule D, section 6). | 1 |
| ELECTORAL_DEFEAT | The governing side loses an election or referendum that is not the scheduled vote installing the next cabinet, and the PM resigns or the government falls as a result. | 1 |
| RESHUFFLE | The same PM, the same parties and the same majority continue, planned or not (including a formal resignation followed by immediate reappointment). | 0 |
| COMPOSITION | Either (a) an unplanned party split where the same ministers stay, the majority is kept and the PM takes no act, or (b) a planned coalition change (enlargement, a negotiated swap of partner). | 0 |
| FORMATION | A new government is formed after a general election or a formation process. | 0 |
| LEADERSHIP_SCHEDULED | A planned succession: term limit, scheduled party leadership contest, a succession set long in advance, or an exit for a post already obtained (rule D). | 0 |
| PRESIDENTIAL | A change required by a scheduled presidential election or inauguration (e.g. the government returns its powers to a new president). | 0 |
| OTHER | An event that fits no code. | ambiguous (always) |

LOST_MAJORITY and COALITION_EXIT overlap (a partner exit that costs the majority fits both); analyses treat them as one category.

A row may deviate from the default only by an explicit decision of the author.

## 4. Trigger-date rules

- **Rule A2 (trigger).** The trigger is the first day a major outlet reports a specific act or statement by an actor who decides the outcome, named or on the record.
  - Anonymous-source reports, analyst assessments and "set to" or "looks like" headlines are not triggers. If seen, they go in `alt_trigger_dates`.
  - A statement by an actor who does not decide the outcome (e.g. the PM's prognosis about a vote others will cast) is not a trigger.
  - Do not search backwards for earlier hints.
  - An actor who decides the outcome is the head of government, the party body that elects or removes the head of government (party room, caucus, directorate), or a partner or support party whose exit removes the majority. Ministers who resign, challengers, and parties whose abstention leaves the majority intact do not decide it (GBR_1676, AUS_1585, ITA_1681).
  - A resignation for health reasons without incapacity follows Rule A2, not the incapacitated-PM rule (JPN_1644).
  - A head of government's on-record commitment to set a timetable for his or her exit is a trigger (GBR_1599).
- **Rule A3 (conditional threats).** A conditional threat or ultimatum ("X unless Y") is not a trigger; it goes in `alt_trigger_dates`. The trigger is the first unconditional on-record act or statement that commits the actor to the outcome.
- **Incapacitated PM.** The trigger is the first day a major outlet reports the hospitalisation; `health_related` = 1. If the coded change is something else and the incapacity is only a confounder, `health_related` stays 0 and the incapacity date goes in alt (ISR_1129).
- **Bundled ParlGov steps.** If one ParlGov step spans several events, code on the cabinet's own dated change. Earlier unplanned events go in `alt_trigger_dates`. If the earlier events are dated on opened pages, the row may enter R_bundle (EST_1688, ISR_1128). ISR_1122 stays out because its earlier exits were seen only in search results.
- **alt_trigger_dates** holds every other dated candidate that is on an opened page, separated by semicolons and in date order.

## 5. Date conventions by event_type

| event_type | trigger_date |
|---|---|
| LOST_MAJORITY, COALITION_EXIT, LEADERSHIP | Rules A2 and A3. |
| ELECTORAL_DEFEAT | The day the result becomes public, in local time. If the opened pages do not give the hour, the vote day is kept. |
| PRESIDENTIAL | The day of the result of the decisive round. |
| FORMATION, planned COMPOSITION | The last party-level approval or signed agreement before installation; if none is on an opened page, the installation day. |
| FORMATION after a direct election of the head of government (ISR_965) | The day the result becomes public. |
| LEADERSHIP_SCHEDULED | The day the successor or the exit date becomes known (on an opened page). |
| RESHUFFLE | The event that prompted the reshuffle, or the reshuffle itself if nothing prompted it. |

`resignation_date` is the day the outgoing PM or cabinet formally resigns, on an opened page. Otherwise it is empty. It stays empty where there is no PM (CHE).

**Day 0 is decided by the analysis code:** it maps `trigger_date` to each index's own trading calendar.

## 6. Rule D (exit of the head of government), as a testable criterion

Apply the tests in order, using only facts on pages opened for the row:

1. **Already scheduled?** Was the date of the exit, or the mechanism that sets it, public before the trigger (term limit, constitutionally required step, a scheduled party leadership contest, an announced succession)? If yes → LEADERSHIP_SCHEDULED, 0.
2. **Post already obtained?** On the trigger date, has the PM already formally obtained the post he or she leaves for (appointment or election to it has happened)? If yes → LEADERSHIP_SCHEDULED, 0. An exit to seek a post not yet obtained fails this test (FIN_1110 → 1).
3. **Otherwise** → LEADERSHIP, 1. This includes forced exits (revolt, scandal, a lost vote) and voluntary but unexpected exits. A voluntary exit with no pressure on the government's position on an opened page also enters R_vol.

## 7. Special cases

1. **A party that splits by itself**, where the same ministers stay, the majority is kept and the PM takes no act, is COMPOSITION 0 (ITA_1068, AUT_888). If the PM acted first to end the partnership, it is COALITION_EXIT 1 (FIN_1510).
2. **A partner leaving on its leader's order**, even when planned or with the government already caretaker, is COALITION_EXIT 1 (ISR_1129).
3. **Constitutionally required re-appointments** after a presidential inauguration are PRESIDENTIAL 0, even when a planned coalition change is bundled into the same step (LTU_1615, LTU_1633).
4. **Switzerland** has no PM and no confidence vote; its change is COMPOSITION 0 (CHE_903).
5. **Party size** comes from ParlGov seats, and COVID overlap is handled in the analysis code, not in coding.

## 8. Robustness sets

Membership is in `robustness_sets.csv`.

- **R_add**: rows coded 0 or ambiguous that are treated as 1 (relabel). This includes every ambiguous row.
- **R_drop**: rows dropped because a simultaneous market-wide shock confounds them: GBR_1489 (the Brexit referendum) and GBR_1683 (market turmoil after the September 2022 mini-budget).
- **R_vol**: voluntary exits without pressure on the government's position (dropped).
- **R_bundle**: rows recoded to 1 on a bundled unplanned event, with a new trigger date (relabel).
- **R_tv**: rows where the choice of date or code is a judgement call (dropped).
- **R_split**: rows recoded to 0 because a governing party split with the same ministers and the same majority. The same split is COMPOSITION 0 in ITA_1068 and AUT_888; for FIN_1510 the only difference is the PM's statement (relabel).
