"""Preprocessing pipeline for AURA inference.

Every function here maps to a specific section of NOTEBOOK_CONTRACT.md.
Changes to these formulas must be reflected in the contract and a fresh
parity fixture must be generated.

# No scaling is applied anywhere in this file.
# The MLP was trained on raw features with no StandardScaler or MinMaxScaler.
# See NOTEBOOK_CONTRACT.md §3.3. Adding scaling here will silently break predictions.
"""

from __future__ import annotations

import math
import re
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.shared.inference.schema import (
    ENGINEERED_FEATURE_ORDER,
    TOTAL_FEATURES,
    URL_PATTERNS,
)


# Pre-compile the combined URL pattern once. Both `normalize_for_tfidf` and
# `count_urls` rely on this.
_COMBINED_URL_PATTERN = re.compile('|'.join(URL_PATTERNS))
_URL_PATTERN_LIST = [re.compile(p) for p in URL_PATTERNS]
_WHITESPACE_PATTERN = re.compile(r'\s+')
_EXCLAIM_QUESTION_PATTERN = re.compile(r'[!?]+')
_SENDER_NAME_EMAIL_PATTERN = re.compile(r'(.+?)\s*<(.+?)>')
_NON_ALPHA_PATTERN = re.compile(r'[^a-zA-Z]')

# NOTEBOOK_CONTRACT §2.3 — shared-inbox local parts that should not be penalised
# by `name_email_consistency`. Legitimate automated senders (GitHub/noreply,
# Coursera/no-reply, Stripe/receipts, etc.) use a display name that does not
# overlap with the mailbox local part; their local part is a generic role
# name, not a personal identifier. Matching is done after stripping non-alpha
# characters from the local part, so 'no-reply' and 'noreply' both hit.
_SHARED_INBOX_LOCALS: frozenset[str] = frozenset({
    'noreply',
    'donotreply',
    'mailerdaemon',
    'postmaster',
    'newsletter',
    'digest',
    'notifications',
    'notification',
    'receipts',
    'support',
    'alerts',
    'hello',
    'contact',
    'info',
    'team',
    'help',
    'news',
    'updates',
    'billing',
    'admin',
})


# --------------------------------------------------------------------------
# NOTEBOOK_CONTRACT §1.B — clean_encoding (data_exploration_cleaning cell 19)
# Minimal pass: HTML entity decode, replacement-char strip, null-byte strip,
# whitespace collapse. No BeautifulSoup. No URL substitution.
# --------------------------------------------------------------------------
def clean_encoding(text: str) -> str:
    # NOTEBOOK_CONTRACT §1 — encoding cleanup
    if text is None:
        return ''
    if isinstance(text, float) and math.isnan(text):
        return ''
    if not isinstance(text, str):
        text = str(text)
    text = text.replace('\ufffd', ' ')
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&amp;', '&')
    text = text.replace('&quot;', '"')
    text = text.replace('\x00', '')
    text = ' '.join(text.split())
    return text


# --------------------------------------------------------------------------
# NOTEBOOK_CONTRACT §2.2 — normalize_text_for_tfidf (feature notebook cell 11)
# Applies only to text fed to the TF-IDF vectorisers, after engineered
# features have already been computed on the cleaned-but-unnormalised text.
# --------------------------------------------------------------------------
def normalize_for_tfidf(text: str) -> str:
    # NOTEBOOK_CONTRACT §2.2 — pre-TF-IDF normalisation
    if text is None:
        return ''
    if isinstance(text, float) and math.isnan(text):
        return ''
    if not isinstance(text, str):
        text = str(text)
    if text.strip() == '':
        return ''
    text = _COMBINED_URL_PATTERN.sub('', text)
    text = _EXCLAIM_QUESTION_PATTERN.sub('', text)
    text = text.lower()
    text = _WHITESPACE_PATTERN.sub(' ', text)
    return text.strip()


# --------------------------------------------------------------------------
# NOTEBOOK_CONTRACT §2.3 — engineered feature primitives (cell 5)
# --------------------------------------------------------------------------
def _extract_sender_components(sender: str) -> tuple[str | None, str | None, str | None]:
    # NOTEBOOK_CONTRACT §2.3 — extract_sender_components
    if sender is None or (isinstance(sender, float) and math.isnan(sender)):
        return None, None, None
    sender = str(sender).strip()
    if sender == '':
        return None, None, None
    match = _SENDER_NAME_EMAIL_PATTERN.search(sender)
    if match:
        sender_name = match.group(1).strip()
        sender_email = match.group(2).strip()
    else:
        sender_name = None
        sender_email = sender
    if sender_email:
        sender_email = sender_email.replace('<', '').replace('>', '').strip()
    if sender_email and '@' in sender_email:
        sender_domain = sender_email.split('@')[-1].strip()
    else:
        sender_domain = None
    return sender_name, sender_email, sender_domain


def _calculate_entropy(text: str) -> float:
    # NOTEBOOK_CONTRACT §2.3 — calculate_entropy (excludes ASCII space)
    if not text or len(text) == 0:
        return 0.0
    freq: dict[str, int] = {}
    for ch in text.lower():
        if ch != ' ':
            freq[ch] = freq.get(ch, 0) + 1
    text_len = sum(1 for c in text if c != ' ')
    if text_len == 0:
        return 0.0
    entropy = 0.0
    for count in freq.values():
        p = count / text_len
        entropy -= p * math.log2(p)
    return entropy


def _email_local_length(email: str | None) -> int:
    # NOTEBOOK_CONTRACT §2.3 — get_email_local_length
    if email is None or email.strip() == '':
        return 0
    if '@' in email:
        return len(email.split('@')[0])
    return 0


def _email_digit_ratio(email: str | None) -> float:
    # NOTEBOOK_CONTRACT §2.3 — get_email_digit_ratio
    if email is None or email.strip() == '':
        return 0.0
    local = email.split('@')[0] if '@' in email else email
    if len(local) == 0:
        return 0.0
    digits = sum(1 for c in local if c.isdigit())
    return digits / len(local)


def _domain_entropy(domain: str | None) -> float:
    # NOTEBOOK_CONTRACT §2.3 — get_domain_entropy (drops TLD when >=2 segments)
    if domain is None or domain.strip() == '':
        return 0.0
    parts = domain.split('.')
    if len(parts) >= 2:
        main = '.'.join(parts[:-1])
    else:
        main = domain
    return _calculate_entropy(main.lower())


def _vowel_consonant_ratio(domain: str | None) -> float:
    # NOTEBOOK_CONTRACT §2.3 — get_vowel_consonant_ratio
    # Vowel set: 'aeiou'. Consonant set: 'bcdfghjklmnpqrstvwxyz' (y=consonant).
    if domain is None or domain.strip() == '':
        return 0.0
    cleaned = _NON_ALPHA_PATTERN.sub('', domain.lower())
    if len(cleaned) == 0:
        return 0.0
    vowels = sum(1 for c in cleaned if c in 'aeiou')
    consonants = sum(1 for c in cleaned if c in 'bcdfghjklmnpqrstvwxyz')
    if consonants == 0:
        return 0.0
    return vowels / consonants


def _name_email_consistency(name: str | None, email: str | None) -> int:
    # NOTEBOOK_CONTRACT §2.3 — check_name_email_consistency (>=3 char threshold)
    if name is None or email is None:
        return 0
    name_clean = _NON_ALPHA_PATTERN.sub('', name.lower())
    email_local = email.split('@')[0] if '@' in email else email
    email_clean = _NON_ALPHA_PATTERN.sub('', email_local.lower())
    if len(name_clean) == 0 or len(email_clean) == 0:
        return 0
    # Shared-inbox whitelist: treat the local part as a role account rather
    # than a personal identifier, so the display-name mismatch is not a phishing
    # signal.
    if email_clean in _SHARED_INBOX_LOCALS:
        return 1
    for part in name_clean.split():
        if len(part) > 2 and part in email_clean:
            return 1
    if len(name_clean) >= 3:
        for i in range(len(name_clean) - 2):
            if name_clean[i:i + 3] in email_clean:
                return 1
    return 0


def _count_urls(text: str) -> int:
    # NOTEBOOK_CONTRACT §2.3 — count_urls (iterates the 15 patterns)
    if text is None or text.strip() == '':
        return 0
    total = 0
    for pattern in _URL_PATTERN_LIST:
        total += len(pattern.findall(text))
    return total


def _avg_word_length(text: str) -> float:
    # NOTEBOOK_CONTRACT §2.3 — get_avg_word_length (uses str.split())
    if text is None or text.strip() == '':
        return 0.0
    words = text.split()
    if len(words) == 0:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def _coerce_text(x) -> str:
    if x is None:
        return ''
    if isinstance(x, float) and math.isnan(x):
        return ''
    return str(x)


# --------------------------------------------------------------------------
# Public: engineered-feature extraction (§2.3, §2.4)
# --------------------------------------------------------------------------
def extract_engineered_features(sender: str, subject: str, body: str) -> np.ndarray:
    # NOTEBOOK_CONTRACT §2.3 + §2.4 — engineered features in training order
    sender_s = _coerce_text(sender)
    subject_s = _coerce_text(subject)
    body_s = _coerce_text(body)

    name, email, domain = _extract_sender_components(sender_s)

    body_word_count = len(body_s.split())
    body_exclamation_count = body_s.count('!')
    email_local_length = _email_local_length(email)
    name_email_consistency = _name_email_consistency(name, email)
    body_url_count = _count_urls(body_s)
    body_url_density = 0.0 if body_word_count == 0 else (body_url_count / body_word_count) * 100
    body_entropy = _calculate_entropy(body_s)
    email_digit_ratio = _email_digit_ratio(email)
    domain_entropy = _domain_entropy(domain)
    domain_length = 0 if domain is None else len(domain)
    subject_entropy = _calculate_entropy(subject_s)
    body_avg_word_length = _avg_word_length(body_s)
    sender_name_exists = 1 if name is not None else 0
    subject_exclamation_count = subject_s.count('!')
    domain_vowel_consonant_ratio = _vowel_consonant_ratio(domain)

    values: dict[str, float] = {
        'body_word_count': float(body_word_count),
        'body_exclamation_count': float(body_exclamation_count),
        'email_local_length': float(email_local_length),
        'name_email_consistency': float(name_email_consistency),
        'body_url_density': float(body_url_density),
        'body_url_count': float(body_url_count),
        'body_entropy': float(body_entropy),
        'email_digit_ratio': float(email_digit_ratio),
        'domain_entropy': float(domain_entropy),
        'domain_length': float(domain_length),
        'subject_entropy': float(subject_entropy),
        'body_avg_word_length': float(body_avg_word_length),
        'sender_name_exists': float(sender_name_exists),
        'subject_exclamation_count': float(subject_exclamation_count),
        'domain_vowel_consonant_ratio': float(domain_vowel_consonant_ratio),
    }
    return np.array([values[name] for name in ENGINEERED_FEATURE_ORDER], dtype=np.float64)


# --------------------------------------------------------------------------
# Public: feature row / batch assembly (§2.4)
# --------------------------------------------------------------------------
def build_feature_row(
    sender: str,
    subject: str,
    body: str,
    subject_vec,
    body_vec,
) -> sp.csr_matrix:
    # NOTEBOOK_CONTRACT §2.4 — assembly order [subject_tfidf | body_tfidf | engineered]
    engineered = extract_engineered_features(sender, subject, body).reshape(1, -1)

    normalized_subject = normalize_for_tfidf(subject)
    normalized_body = normalize_for_tfidf(body)

    subj_tfidf = subject_vec.transform([normalized_subject])
    body_tfidf = body_vec.transform([normalized_body])

    row = sp.hstack([subj_tfidf, body_tfidf, sp.csr_matrix(engineered)], format='csr')
    if row.shape != (1, TOTAL_FEATURES):
        raise ValueError(
            f'Assembled row has shape {row.shape}, expected (1, {TOTAL_FEATURES}). '
            'Check vectoriser feature dims.'
        )
    return row


def build_feature_matrix(
    emails: Iterable[dict],
    subject_vec,
    body_vec,
) -> sp.csr_matrix:
    # NOTEBOOK_CONTRACT §2.4 — vectorised batch path
    emails_list = list(emails)
    if len(emails_list) == 0:
        return sp.csr_matrix((0, TOTAL_FEATURES), dtype=np.float64)

    senders = [e.get('sender', '') for e in emails_list]
    subjects = [e.get('subject', '') for e in emails_list]
    bodies = [e.get('body', '') for e in emails_list]

    engineered = np.vstack([
        extract_engineered_features(s, sub, b)
        for s, sub, b in zip(senders, subjects, bodies)
    ])

    normalized_subjects = [normalize_for_tfidf(s) for s in subjects]
    normalized_bodies = [normalize_for_tfidf(b) for b in bodies]

    subj_tfidf = subject_vec.transform(normalized_subjects)
    body_tfidf = body_vec.transform(normalized_bodies)

    matrix = sp.hstack([subj_tfidf, body_tfidf, sp.csr_matrix(engineered)], format='csr')
    if matrix.shape != (len(emails_list), TOTAL_FEATURES):
        raise ValueError(
            f'Feature matrix has shape {matrix.shape}, expected '
            f'({len(emails_list)}, {TOTAL_FEATURES}).'
        )
    return matrix
