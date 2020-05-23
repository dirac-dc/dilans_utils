import functools
from itertools import chain
import re
import warnings
from typing import List, Dict, Iterable, Tuple, Callable, Optional

import unidecode
import numpy as np
import pandas as pd

from .company_text_data import COMPANY_KINDS_MAP, RECIPIENT, PUNCS, SPACE_PUNCS, BLACKLIST_DOMAINS


@functools.lru_cache(1)
def prepare_names(series_function: Callable[['CompanyCleaner', pd.Series], pd.Series]):
    """Decorate bound methods of CompanyCleaner with some string cleaning before their unique purpose."""
    def inner(self, series: pd.Series) -> pd.Series:
        series = series.fillna("")
        if self.drop_invalid_utf8:
            series = series.str.encode('utf-8', 'ignore').str.decode('utf-8', 'ignore')
        if self.map_to_ascii:
            series = series.map(unidecode.unidecode)  # can introduce capitalisation
        return series_function(self, series.str.lower())

    return inner


class NameCleaner:
    # Allowed characters of a domain name (BETWEEN dos).
    STEM_RANGE = r'[a-zA-Z0-9\-]'
    DOMAIN_RANGE = r'[a-zA-Z]'

    EMAIL_PREFIX = r'\w@'
    EMAIL_STEM = '({stem_range}+)'.format(stem_range=STEM_RANGE)
    EMAIL_DOMAINS = r'\.\w'

    # Totally optional starting sting such as https://www.
    WEBSITE_PREFIX = r'(?:https?)?(?::\/\/)?(?:www\.)?'

    MATCH_NOTHING = r'a^'

    REPLACE_ANDS = ((r'\band\b', ' and '), ('+', ' and '), ('&', ' and '))
    TRADING_AS = r'((\bt/?a\b)|trading as)'

    def __init__(self, remove_company_kind=True, remove_punctuation=True, remove_brackets=True,
                 remove_square_brackets=True, remove_recipient=True, drop_invalid_utf8=True, map_to_ascii=True,
                 domain_extract='stem', clean_emails=True, clean_websites=True, finalize_whitespace='normalise',
                 replace_pairs=((r'[0-9]{4,}', ''),), homogenise_ands=False, ta_handling='left', extract_uid_pat=None,
                 space_puncs=SPACE_PUNCS, domain_punctuation='abolish', company_kinds_map=COMPANY_KINDS_MAP['default'],
                 short_company_kind_start=False, recipient=RECIPIENT['default'], puncs=PUNCS,
                 blacklist_domains=BLACKLIST_DOMAINS):
        """Clean company names.
        Note on anchored regex
        ----------------------
        In this application of regex, there are many situations where it is useful to anchor a pattern to the end of
        the string like so: r"pattern$". However, a single string might end in several pieces of junk, and by using
        the $ character you will miss all but the last of them. Therefore, in this class, patterns ending in $ or
        starting with ^ are combined like so:
        input = [r'pattern1$', r'pattern2$']
        output = r'(?:pattern1|pattern2)+$'
        See `self.compile_patterns` and its callers for the mechanics of this.
        A downside of this is that a pattern is able to repeatedly consume the string it is matching. For example,
        if you pass an `extract_uid_pat` of `^.{4}`, four characters will be consumed repeatedly. To stop this,
        and truly anchor your pattern, just pass it in brackets: '(^.{4})`.
        This note affects `extract_uid_pat`, company kinds, and all the stuff defined in `general_abolish_regex`. It
        doesn't affect `replace_pairs` or punctuation.
        Parameters
        ----------
        remove_company_kind : bool
            Remove legal company identifiers defined in `self.LEGAL_IDENTIFIERS`. This happens after punctuation
            cleaning, so its effects will depend upon what is done with punctuation.
        remove_punctuation : bool
            Remove punctuation defined in `self.PUNCTUATION` and `self.PUNC_SPECIALS`.
        remove_brackets : bool
            Remove all brackets and their contents
        remove_recipient : bool
            Remove miscellaneous recipient patterns that aren't part of the company name. See `self.MISC_RECIPIENT`.
        finalize_whitespace : {None, 'abolish', 'normalise'}
            Action on whitespace characters '\\s+' at the end of the cleaning process. With None, take no action.
            With 'abolish', replace with the empty string. With 'normalize', replace with ' '.
        space_puncs : list of str
            These strings will be replaced by a space instead of nothing. Make sure they are escaped if they are
            regex special characters (see `PUNC_SPECIALS`). This option has no effect if `remove_punctuation` is False.
        domain_extract : {'stem', 'domain', 'all'}
            Determines what to extract from emails and websites when `self.email_extract` and `self.website_extract`
            are called. Let's take person@clear.ai and https://website.co.uk as examples.
            - 'stem' gives you 'clear' and 'website'
            - 'domain' gives you 'clear.ai' and 'website.co.uk'
            - 'all' gives you 'person@clear.ai' and 'https://website.co.uk'
            Note that the regex for emails is written to allow for other junk around the email address which will not
            stop the email being identified. The regex for websites is written more strictly and a website will not
            be recognised if there is junk around it.
        clean_emails : bool
            Clean emails as part of the main cleaning process (follows `domain_extract`).
        clean_websites : bool
            Clean websites as part of the main cleaning process (follows `domain_extract`).
        drop_invalid_utf8 : bool
            Use `Series.str.encode('utf-8', 'ignore').str.decode('utf-8', 'ignore')` to drop invalid utf-8 codes.
            This affects both the `.clean` method and the `.extract_...` methods.
        map_to_ascii : bool
            Map all names to ascii during the cleaning process and before all extractions (like company kind).
            Even works for foreign scripts (有 is 'You'), but perhaps most useful for normalizing the Latin alphabet.
            This affects both the `.clean` method and the `.extract_...` methods.
        replace_pairs : iterable of 2-tuples of regex strings
            Arbitrary regex replacements to execute on the names that are not websites or emails.
            These are the first actions taken after `prepare_names`.
        homogenise_ands : bool
            Adds the following tuples to the `replace_pairs` parameter:
            ('\band\b', ' and '), ('+', ' and '), ('&', ' and ')
        ta_handling : {None, 'left', 'right'}
            If None, no effect. If 'left' or 'right', this has the effect of replacing the entire string with everything
            found to the left or right of the substring matching `self.TRADING_AS`. If 'remove', the regex
            `self.TRADING_AS` is removed from each line. This works by adding to `replace_pairs`.
            'left' is the recommended setting for matching to an official dataset of company names.
        extract_uid_pat : regex string
            Pattern to be removed from the name during cleaning, and extracted by the `extract_uid` function. Should
            expect dirty company names. This is intended to be a string that uniquely represents the company.
        domain_punctuation: {'abolish', 'leave', 'same'}
            What to do with punctuation found when stemming emails and websites. 'abolish' will remove all punctuation.
            'leave' will not touch punctuation. 'same' will apply the same rules as have been set for normal strings.
            This argument only has an effect if `(clean_emails | clean_website)`.
        company_kinds_map : dict-like
            For functionality concerned with removing/extracting company kinds, this allows passing a custom
            dict of company strings to look for (keys) and what they should be labelled as in `extract_company_kind`.
        short_company_kind_start : bool
            As the logic in `build_company_kind_pats` shows, company kinds (keys of the `company_kinds_map` parameter)
            which are shorter than 4 characters are only matched if they occur at the end of the company name. If this
            parameter is set to True, matches will be accepted also if they occur at the start of the company name.
        recipient: list of str
            Sender or receiver information that is not the company name.
        puncs : list of str
            List of characters to consider as punctuation for cleaning. Must be properly escaped. The default should
            normally suffice.
        blacklist_domains : list of str
            Names to replace with `np.nan` if they are found from stemming the emails or websites.
        """
        self.remove_punctuation = remove_punctuation
        self.remove_brackets = remove_brackets
        self.remove_square_brackets = remove_square_brackets
        self.remove_recipient = remove_recipient
        self.remove_company_kind = remove_company_kind

        self.map_to_ascii = map_to_ascii
        self.drop_invalid_utf8 = drop_invalid_utf8

        self.replace_pairs = replace_pairs
        if homogenise_ands:
            self.replace_pairs = self.replace_pairs + self.REPLACE_ANDS

        if ta_handling == 'left':
            self.replace_pairs = self.replace_pairs + ((self.TRADING_AS + r'.*', r''),)
        elif ta_handling == 'right':
            self.replace_pairs = self.replace_pairs + ((r'.*' + self.TRADING_AS, r''),)
        elif ta_handling == 'remove':
            self.replace_pairs = self.replace_pairs + ((self.TRADING_AS, ''),)

        self.company_kind_pats, self.company_kind_labels = self.build_company_kind_pats(company_kinds_map,
                                                                                        short_company_kind_start)

        self.recipient = recipient

        self.extract_uid_pat = extract_uid_pat if extract_uid_pat is not None else '({})'.format(self.MATCH_NOTHING)

        self.blacklist_domains = blacklist_domains

        first_stem_pat = r'(?:{range}{{2,}})'.format(range=self.STEM_RANGE)
        second_stem_pat = r'(?:\.{range}{{2,}})*'.format(range=self.STEM_RANGE)
        outer_domain_pat = r'(?:\.{range}{{2,}})+'.format(range=self.DOMAIN_RANGE)
        terminal_pat = r'\/?$'
        email_prefix = r'[a-zA-Z0-9_\-\.]+'

        if domain_extract == 'stem':
            self.website_extract_re = (r'^'
                                       + self.WEBSITE_PREFIX
                                       + '(' + first_stem_pat + ')'
                                       + second_stem_pat
                                       + outer_domain_pat
                                       + terminal_pat)
            self.email_extract_re = (email_prefix
                                     + r'\@'
                                     + '(' + first_stem_pat + ')'
                                     + second_stem_pat
                                     + outer_domain_pat)

        elif domain_extract == 'domain':
            self.website_extract_re = (r'^'
                                       + self.WEBSITE_PREFIX
                                       + r'(' + first_stem_pat
                                       + second_stem_pat
                                       + outer_domain_pat + ')'
                                       + terminal_pat)
            self.email_extract_re = (email_prefix
                                     + r'\@'
                                     + '(' + first_stem_pat
                                     + second_stem_pat
                                     + outer_domain_pat + ')')

        elif domain_extract == 'all':
            self.website_extract_re = (r'^('
                                       + self.WEBSITE_PREFIX
                                       + first_stem_pat
                                       + second_stem_pat
                                       + outer_domain_pat + ')'
                                       + terminal_pat)
            self.email_extract_re = ('(' + email_prefix
                                     + r'\@'
                                     + first_stem_pat
                                     + second_stem_pat
                                     + outer_domain_pat + ')')
        else:
            raise ValueError("Unrecognised value for `domain_extract`: {}".format(domain_extract))

        self.email_func = self.email_extract if clean_emails else lambda series: series
        self.website_func = self.website_extract if clean_websites else lambda series: series

        self.domain_punctuation = domain_punctuation

        self.abolish_puncs = [c for c in puncs if c not in space_puncs]
        self.space_puncs = space_puncs
        self.abolish_puncs_regex = self.compile_patterns(self.abolish_puncs, anchor=None)
        self.space_puncs_regex = self.compile_patterns(self.space_puncs, anchor=None)

        self.abolish_patterns = {
            'uid': [self.extract_uid_pat],
            'company kind': self.company_kind_pats if self.remove_company_kind else [self.MATCH_NOTHING],
            'general': self.general_abolish_regex(),
        }

        self.finalize_whitespace = finalize_whitespace

    @staticmethod
    def compile_patterns(patterns: List[str], anchor: Optional[str]):
        """Turn a list of patterns into a non-matching group separated by | (representing OR).
        Parameters
        ----------
        patterns : List[str]
            Regex patterns to be compiled into a group
        anchor : {'start', 'end', None}
            Assume input starts [ends] with '^' ['$']. Remove these characters and instead collectively anchor the whole
            list of patterns to the start [end] of the string being matched. Please see "Note on anchored regex" in
            __init__'s docstring for more information.
        Returns
        -------
        re.Pattern
            Compiled regex pattern.
        """
        start = ending = ''
        if anchor == 'start':
            patterns = [pattern[1:] for pattern in patterns]
            start = '^'
        elif anchor == 'end':
            patterns = [pattern[:-1] for pattern in patterns]
            ending = '$'

        if patterns:
            core = '|'.join(patterns)
        else:
            core = CompanyCleaner.MATCH_NOTHING  # If iter is empty, return regex that can match nothing.

        return re.compile(start + '(?:' + core + ')+' + ending)

    @staticmethod
    def build_company_kind_pats(company_kinds_map: Dict[str, str], short_company_kind_start: bool):
        """Build the company kinds (ltd, corp., s/a etc) regex and the list of their values for extraction
        Company kinds can often occur with variable characters in-between, such as [' ', '.', '.', '/'], so this is
        built into the regex. Also, since these acronyms might occur elsewhere in the string by coincidence, we force
        company kinds shorter than 4 characters to be anchored to the start or end of the company name.
        (Though, see Note on anchored regex in __init__'s doc.)
        """
        spacing_pat = r"[ \.,\/\\]?"
        company_kinds = sorted(company_kinds_map.keys(), key=len, reverse=True)
        patterns, replacements = [], []

        for company_kind in company_kinds:
            is_short = len(company_kind) < 4
            replacement = company_kinds_map[company_kind]

            pattern = r"\b" + spacing_pat.join(company_kind.lower()) + spacing_pat
            end = r'\W*$' if is_short else r'\b'

            patterns.append(pattern + end)
            replacements.append(replacement)

            if short_company_kind_start and is_short:
                patterns.append(r"^\W*" + pattern + r"\b")
                replacements.append(replacement)

        return patterns, replacements

    def general_abolish_regex(self) -> List[str]:
        """Create the regex pattern with an OR pipe separating each possible character or pattern to eliminate.
        The regex patterns here are the ones we will always want to use during any extraction or cleaning.
        """
        abolish_patterns = []

        if self.remove_brackets:
            abolish_patterns += [r'\((.*?)\)']

        if self.remove_square_brackets:
            abolish_patterns += [r'\[(.*?)\]']

        if self.remove_recipient:
            abolish_patterns += self.recipient

        return abolish_patterns

    @staticmethod
    def _replace_from_pairs(raw_names: pd.Series, pairs: Tuple) -> pd.Series:
        for match, replace in pairs:
            raw_names = raw_names.str.replace(match, replace)

        return raw_names

    @prepare_names
    def clean(self, raw_names: pd.Series) -> pd.Series:
        """Split into emails, websites, and everything else. Clean each separately and bring back together.
        Parameters
        ----------
        raw_names : pd.Series
            Column of strings to transform
        Returns
        -------
        out : pd.Series
            Cleaned column of strings.
        """
        name_patterns = chain(*self.abolish_patterns.values())

        is_email = self.is_email_mask(raw_names)
        is_website = self.is_website_mask(raw_names)
        is_name = ~(is_email | is_website)

        df_email = self.email_func(raw_names[is_email])
        df_website = self.website_func(raw_names[is_website])
        df_name = raw_names[is_name].pipe(self._clean_name, name_patterns).pipe(self._finalise_name)

        return raw_names.mask(is_name, df_name).mask(is_email, df_email).mask(is_website, df_website).str.strip()

    @prepare_names
    def extract_company_kind(self, raw_names: pd.Series) -> pd.Series:
        """Extract the company kinds from `self.COMPANY_KINDS_MAP` into their own column."""

        kind = pd.Series(np.nan, index=raw_names.index)
        kind[self.is_email_mask(raw_names)] = 'email'
        kind[self.is_website_mask(raw_names)] = 'website'

        # Clean names up to company kinds. This increases the chance of matching anchored patterns '$' or '^'
        # Note that we do not remove punctuation or finalise the whitespace.
        name_patterns = [patterns for kind, patterns in self.abolish_patterns.items() if kind != 'company kind']
        cleaned_names = self._clean_name(raw_names, chain(*name_patterns))

        # We don't take the company kinds patterns from `abolish_patterns['company kind']` but rather from
        # self.company_kind_pats. This is because the former won't be set if the user doesn't want it cleaned.
        for pattern, label in zip(self.company_kind_pats, self.company_kind_labels):
            kind[kind.isnull() & cleaned_names.str.contains(pattern)] = label

        return kind

    @prepare_names
    def email_extract(self, email_series: pd.Series):
        """Get the domain name from an email address. 'a.person@email.com' becomes 'email'."""
        return (email_series
                .str.extract(self.email_extract_re, expand=False)
                .map(lambda domain: '' if domain in self.blacklist_domains else domain)
                .pipe(self._clean_domain_punctuation))

    @prepare_names
    def is_email_mask(self, names: pd.Series) -> pd.Series:
        """Simple case-insensitive email detection."""
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message="This pattern has match groups. To actually get the group")
            return names.str.contains(self.email_extract_re)

    @prepare_names
    def website_extract(self, website_series: pd.Series) -> pd.Series:
        """Get the lowest-level domain name from a website. 'www.example.com.ar' becomes 'example'."""
        return (website_series
                .str.extract(self.website_extract_re, expand=False)
                .map(lambda domain: '' if domain in self.blacklist_domains else domain)
                .pipe(self._clean_domain_punctuation))

    def _clean_domain_punctuation(self, domain_series: pd.Series) -> pd.Series:
        if self.domain_punctuation == 'abolish':
            domain_series = (domain_series.str.replace(self.space_puncs_regex, '')
                             .str.replace(self.abolish_puncs_regex, ''))
        elif self.domain_punctuation == 'same' and self.remove_punctuation:
            domain_series = (domain_series.str.replace(self.space_puncs_regex, ' ')
                             .str.replace(self.abolish_puncs_regex, ''))

        return domain_series

    @prepare_names
    def extract_uid(self, raw_names: pd.Series) -> pd.Series:
        """Extract the company uids matching `self.extract_uid_pat` into their own column."""
        name_patterns = (patterns for kind, patterns in self.abolish_patterns.items() if kind != 'uid')
        cleaned_names = self._clean_name(raw_names, chain(*name_patterns))
        return cleaned_names.str.extract(self.extract_uid_pat, expand=False)

    @prepare_names
    def is_website_mask(self, names: pd.Series) -> pd.Series:
        """Website detection."""
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message="This pattern has match groups. To actually get the group")
            return names.str.contains(self.website_extract_re)

    def _clean_name(self, raw_names: pd.Series, abolish_patterns: Iterable[str]) -> pd.Series:
        """Transformation route for rows that are not found to be emails or website."""
        floating, anchor_start, anchor_end = self._compile_abolish_regexes(abolish_patterns)
        return (raw_names
                .pipe(self._replace_from_pairs, pairs=self.replace_pairs)
                .str.replace(floating, '')  # Floating matches might get in the way of anchored matches.
                .str.replace(anchor_start, '')
                .str.replace(anchor_end, ''))

    def _compile_abolish_regexes(self, abolish_patterns: Iterable[str]):
        """Compile regexes into three groups by anchor pattern: '^', '$', or neither.
        '^' and '$' groups are combined in such a way that multiple 'terminating' patterns can be found in one string
        """
        floating, anchor_start, anchor_end = [], [], []

        for pattern in abolish_patterns:
            if pattern[0] == '^':
                anchor_start.append(pattern)
            elif pattern[-1] == '$' and pattern[-2] != '\\':
                anchor_end.append(pattern)
            else:
                floating.append(pattern)

        return (self.compile_patterns(floating, anchor=None),
                self.compile_patterns(anchor_start, anchor='start'),
                self.compile_patterns(anchor_end, anchor='end'))

    def _finalise_name(self, clean_names: pd.Series) -> pd.Series:
        """Remove punctuation and whitespace according to settings."""
        if self.remove_punctuation:
            clean_names = (clean_names.str.replace(self.space_puncs_regex, ' ')
                           .str.replace(self.abolish_puncs_regex, ''))
        if self.finalize_whitespace == 'abolish':
            clean_names = clean_names.str.replace(r'\s+', '')
        elif self.finalize_whitespace == 'normalise':
            clean_names = clean_names.str.replace(r'\s+', ' ')

        return clean_names
