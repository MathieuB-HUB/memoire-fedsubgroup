# Preenregistrement des hypotheses

Document date a la creation, AVANT de regarder les resultats. Sert de garde-fou anti p-hacking.

Date de preenregistrement : 2026-05-24.

## Cadre

Comparaison de 6 methodes d'apprentissage federe (FedAvg, FedBN, FedPer, FedRep, Ditto, FedSubgroup) sur deux datasets de sante (PTB-XL, Fed-Heart-Disease), avec deux attributs sensibles (sexe, age), 3 seeds.

Metriques principales :
- Performance globale : AUC macro
- Equite : worst-group AUC, gap min-max
- Equite (secondaire) : demographic parity gap, equalized odds gap

Test statistique : Wilcoxon apparie sur les 3 seeds. IC bootstrap (1000 resamples) sur worst-group AUC.

## Hypotheses

### H1 - FedSubgroup ameliore l'equite vs personnalisation par client

FedSubgroup obtient un worst-group AUC strictement superieur a la meilleure des methodes de la philosophie B (FedBN, FedPer, FedRep, Ditto), sur au moins un des deux datasets, pour au moins un des deux attributs sensibles, avec p < 0.05 (Wilcoxon).

### H2 - FedSubgroup egale ou bat la perso par client sur la performance globale

L'AUC macro de FedSubgroup est superieur ou egal (delta < 0.01) a la meilleure methode de la philosophie B, sur les memes conditions que H1.

### H3 - Compromis explicite

Si H1 confirmee mais H2 non : FedSubgroup ameliore l'equite au prix d'une legere baisse de performance moyenne. Le compromis est quantifie : combien de points d'AUC global perdus par point de worst-group AUC gagne.

### H4 - Resultat negatif possible

FedSubgroup n'apporte pas d'amelioration significative vs philosophie B. Interpretation : la personnalisation par lieu capte deja implicitement la demographie locale en FL sante. Resultat publiable en soi.

## Engagement

Quelle que soit l'hypothese confirmee, elle sera reportee honnetement dans le memoire. Pas de reformulation a posteriori.
