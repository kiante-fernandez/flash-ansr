"""simplipy 0.14 removed ``SimpliPyEngine.parse``. These tests MEASURE what the two
call sites that used it would become under the replacement surface
(``to_prefix`` + ``engine.mask``), over the real corpora, and pin the result.

The verdict they record: NEITHER site can be migrated byte-identically, so neither is
migrated. ``convert_data.py`` and ``flash_ansr_model.py`` still call ``engine.parse``
and are BLOCKED pending an owner decision -- the divergences are training-data /
candidate-set semantics, not spelling.

WHY THE OLD BEHAVIOUR IS STILL REACHABLE HERE. The removed ``engine.parse`` was a pure
delegation to the Rust raw reader (``self._core.parse(infix, convert_expression,
mask_numbers)``); simplipy 0.14 keeps that reader as the conversion hub's INTERNAL entry
(no compatibility promise). These tests call it as the ORACLE for the removed public
method -- it is the only way to compare the old and new spellings in one process, and
its subject IS the removed behaviour. Production code must not use it.

THE THREE AXES on which ``parse(s, mask_numbers=True)`` and ``mask(to_prefix(s))`` differ:

1. POLICY -- ``mask_numbers=True`` ran ``numbers_to_constant``: replace every token for
   which Python's ``float(token)`` succeeds. That is ``masking.mask_all`` on every role
   INCLUDING ``pow`` exponents and ``rootn`` indices, but EXCLUDING ``np.pi`` / ``np.e``
   (``float('np.pi')`` raises, while ``mask_all`` masks the special constants by ruling).
   ``mask_fittable`` -- and its alias ``mask_values_keep_structure``, the SAME function
   object -- keeps exponents and root indices, so it is a different mask entirely.
2. COLLECT -- the toolkit's stage (b) re-runs the engine over the masked tokens
   (one ``<constant>`` per degree of freedom). ``engine.mask`` always collects; the old
   spelling was positional only.
3. CANONICALISATION -- the removed reader was spelling-preserving; ``to_prefix`` reads
   into the canonical state, so it masks a DIFFERENT expression: literals fold before
   masking, ``1/u`` becomes ``inv u``, ``2.5`` becomes the exact rational, and an
   out-of-vocabulary function name (``sqrt``) is REFUSED where the raw reader passed it
   through as a leaf.
"""
import os

import pytest
import yaml
from simplipy import SimpliPyEngine, masking

from flash_ansr.utils.skeleton import mask_all_literals, simplify_and_mask
from flash_ansr import get_path


FASTSRB = get_path('data', 'ansr-data', 'test_set', 'fastsrb', 'expressions.yaml')

# Infix spellings the FastSRB `prepared` column cannot exercise (it spells pi as the
# decimal 3.1415926535897), each discriminating one policy axis.
DISCRIMINATING = [
    'pi * x1 + 2',
    'e ** x1',
    'pi * e * x1',
    'x1 ** 3 + 2 * x1',
    'x1 ** 2 / 3',
    '2.5 * x1 + 0.5',
    '1e-05 * x1',
    '-2 * x1 - 3',
    'log(2 * x1) + exp(0.5 * x1)',
    'pi * x1 ** 2 + 2.5 * x2',
]

# Real `simplified_infix` values at the flash_ansr_model.py post-processing site, captured
# from the site's own pipeline: beam tokens -> prefix_to_infix(power='**') -> sympy.simplify
# -> Abs->abs. Frozen so the test needs neither sympy nor a trained model.
SYMPY_OUTPUTS_AGREEING = [
    '2/(x1*log(2))', 'x1 + x2*sin(x1)', 'cos(2/x0)', '0.5**(0.5/x1**2)',
    '33.1154519586923', '-0.500000000000000', '-4/(x1 - x2)', 'tanh(exp(sin(3)))',
    '3.50000000000000', 'atan(sin(x1))',
]
# ... and real ones where the two spellings disagree, with both artifacts.
SYMPY_OUTPUTS_DIVERGING = [
    ('sin(3*x1)**0.5',
     ['pow', 'sin', '*', '<constant>', 'x1', '<constant>'],
     ['rootn', 'sin', '*', '<constant>', 'x1', '<constant>']),
    ('1/x0',
     ['/', '<constant>', 'x0'],
     ['inv', 'x0']),
    ('x0/2 + x1/2',
     ['+', '*', '<constant>', 'x0', '*', '<constant>', 'x1'],
     ['*', '<constant>', '+', 'x0', 'x1']),
    ('-x0 + x1',
     ['+', 'neg', 'x0', 'x1'],
     ['-', 'x1', 'x0']),
    ('log(x2**(-3))',
     ['log', 'pow', 'x2', '<constant>'],
     ['log', 'inv', 'pow', 'x2', '<constant>']),
    ('3*x2/atan(2) + 1/x1',
     ['+', '*', '<constant>', 'x2', '/', '<constant>', 'x1'],
     ['+', 'inv', 'x1', '*', '<constant>', 'x2']),
]


@pytest.fixture(scope='module')
def engine() -> SimpliPyEngine:
    return SimpliPyEngine.load('acj-4-3', install=True)


@pytest.fixture(scope='module')
def fastsrb_corpus() -> list[tuple[str, str]]:
    """The real convert_data corpus: FastSRB `prepared` expressions, with the FastSRBParser's
    own ``^`` -> ``**`` normalisation applied."""
    if not os.path.exists(FASTSRB):
        pytest.skip('FastSRB benchmark fixture not downloaded')
    with open(FASTSRB) as file:
        document = yaml.safe_load(file)
    return [(eq_id, entry['prepared'].replace('^', '**'))
            for eq_id, entry in document.items()
            if isinstance(entry.get('prepared'), str) and entry['prepared'].strip()]


def old_parse(engine: SimpliPyEngine, expression: str, mask_numbers: bool = False) -> list[str]:
    """The removed ``engine.parse``, verbatim: it delegated to this reader and nothing else."""
    return engine._core.parse(expression, True, mask_numbers)


def float_predicate(value: str, role: masking.Role) -> str | None:
    """``numbers_to_constant``'s predicate as a masking policy: mask iff ``float(token)`` works."""
    try:
        float(value)
    except ValueError:
        return None
    return '<constant>'


# --------------------------------------------------------------------------------------
# AXIS 1: which shipped policy IS mask_numbers=True?
# --------------------------------------------------------------------------------------

def test_mask_fittable_and_mask_values_keep_structure_are_one_function() -> None:
    """Three names, two behaviours: 'values' is an alias, so the choice is all-vs-fittable."""
    assert masking.mask_fittable is masking.mask_values_keep_structure


def test_mask_numbers_true_is_positional_mask_all_except_the_special_constants(engine: SimpliPyEngine, fastsrb_corpus: list[tuple[str, str]]) -> None:
    """MAPPING PROOF, over the real corpus + the discriminating spellings.

    ``mask_numbers=True`` == ``masking.mask(..., collect=False)`` under the float()
    predicate, everywhere the role walk can run. ``mask_all`` agrees with it on every
    expression that has no ``np.pi`` / ``np.e``, and disagrees on every one that has.
    """
    corpus = [expression for _, expression in fastsrb_corpus] + DISCRIMINATING
    checked = specials = 0

    for expression in corpus:
        raw = old_parse(engine, expression)
        expected = old_parse(engine, expression, mask_numbers=True)
        try:
            by_predicate = masking.mask(list(raw), engine, float_predicate, collect=False)
        except ValueError:
            # the role walk refuses out-of-vocabulary names (`sqrt`); covered by its own test
            continue
        checked += 1
        assert by_predicate == expected, expression

        by_mask_all = masking.mask(list(raw), engine, masking.mask_all, collect=False)
        if 'np.pi' in raw or 'np.e' in raw:
            specials += 1
            assert by_mask_all != expected, expression
        else:
            assert by_mask_all == expected, expression

    assert checked > 90, 'corpus too small to be a proof'
    assert specials >= 4, 'corpus does not exercise the np.pi / np.e difference'


def test_mask_numbers_true_is_not_mask_fittable(engine: SimpliPyEngine) -> None:
    """``fittable``/``values`` KEEP what ``mask_numbers=True`` masked: the structural
    exponent and the root index."""
    raw = old_parse(engine, 'x1 ** 3 + 2 * x1')
    assert old_parse(engine, 'x1 ** 3 + 2 * x1', mask_numbers=True) == [
        '+', 'pow', 'x1', '<constant>', '*', '<constant>', 'x1']
    assert masking.mask(list(raw), engine, masking.mask_fittable, collect=False) == [
        '+', 'pow', 'x1', '3', '*', '<constant>', 'x1']


def test_engine_mask_always_collects_so_it_cannot_reproduce_the_old_bytes(engine: SimpliPyEngine) -> None:
    """AXIS 2. ``engine.mask`` has no ``collect=False`` escape, and the collect stage
    re-orders and re-shapes: ``2 * (1 + v2)`` masks to a different token order."""
    expression = 'v1 / (2 * (1 + v2))'
    assert old_parse(engine, expression, mask_numbers=True) == [
        '/', 'v1', '*', '<constant>', '+', '<constant>', 'v2']
    assert engine.mask(old_parse(engine, expression), policy='all') == [
        '/', 'v1', '*', '<constant>', '+', 'v2', '<constant>']


# --------------------------------------------------------------------------------------
# AXIS 3 / the convert_data blocker
# --------------------------------------------------------------------------------------

def test_to_prefix_refuses_the_out_of_vocabulary_names_the_raw_reader_passed_through(engine: SimpliPyEngine) -> None:
    """BLOCKER CLASS 1 (26 of the 120 FastSRB entries). The raw reader passed ``sqrt``
    through as a leaf, so the site's ``is_valid`` gate counted the entry as invalid and
    SKIPPED it (designed attrition). ``to_prefix`` raises instead, which at
    ``_process_expression``'s default ``skip_unparseable=False`` aborts the whole import.
    """
    passed_through = old_parse(engine, 'sqrt(v1 * v2 / v3)')
    assert passed_through == ['sqrt', '/', '*', 'v1', 'v2', 'v3']
    assert engine.is_valid(passed_through) is False

    with pytest.raises(ValueError):
        engine.to_prefix('sqrt(v1 * v2 / v3)')


def test_convert_data_migration_changes_the_number_of_free_constants(engine: SimpliPyEngine) -> None:
    """BLOCKER CLASS 2. Masking BEFORE canonicalisation (old) and AFTER it (new) are not
    the same skeleton: canonicalisation folds the literals the old spelling had already
    turned into independent ``<constant>``s, and drops the ones that were structure.

    Each degree of freedom is a parameter the refiner fits, so this is a training-data
    semantics change, not a spelling difference.
    """
    cases = [
        # (expression, old constants, new constants)
        ('1 / (4 * 3.1415926535897 * 8.854e-12 * 2.99792458e8 ** 2) * 2 * v1 / v2', 3, 1),
        ('6.67430e-11 * v1 * v2 * (1 / v3 - 1 / v4)', 3, 1),
        ('1 / (1 / v1 + v2 / v3)', 2, 0),
    ]
    for expression, n_old, n_new in cases:
        old_artifact = simplify_and_mask(engine, old_parse(engine, expression, mask_numbers=True))
        new_artifact = simplify_and_mask(engine, engine.mask(engine.to_prefix(expression), policy='all'))
        assert old_artifact != new_artifact, expression
        assert old_artifact.count('<constant>') == n_old, (expression, old_artifact)
        assert new_artifact.count('<constant>') == n_new, (expression, new_artifact)


@pytest.mark.xfail(strict=True, reason='BLOCKED: the migration is not artifact-preserving; '
                                       'this is the acceptance criterion, red until the owner rules')
def test_convert_data_artifact_is_byte_identical_over_the_fastsrb_corpus(engine: SimpliPyEngine, fastsrb_corpus: list[tuple[str, str]]) -> None:
    """The acceptance criterion for migrating ``convert_data._process_expression``: every
    FastSRB entry's imported skeleton unchanged. Measured 2026-08-18 against the Part-A
    build: 26 entries raise, 20 more produce a different skeleton, 74 agree."""
    for eq_id, expression in fastsrb_corpus:
        old_tokens = old_parse(engine, expression, mask_numbers=True)
        new_tokens = engine.mask(engine.to_prefix(expression), policy='all')
        if not engine.is_valid(old_tokens):
            continue
        assert simplify_and_mask(engine, old_tokens) == simplify_and_mask(engine, new_tokens), eq_id


# --------------------------------------------------------------------------------------
# The flash_ansr_model.py sympy-branch blocker
# --------------------------------------------------------------------------------------

def test_model_sympy_branch_agrees_on_the_already_canonical_outputs(engine: SimpliPyEngine) -> None:
    """Where SymPy's printer happens to spell the canonical state, the two spellings do
    produce the same candidate -- this is the majority (measured 356 of 400 real outputs)."""
    for expression in SYMPY_OUTPUTS_AGREEING:
        old_tokens = mask_all_literals(engine, old_parse(engine, expression))
        new_tokens = mask_all_literals(engine, engine.to_prefix(expression))
        assert old_tokens == new_tokens, expression


def test_model_sympy_branch_diverges_on_real_sympy_output(engine: SimpliPyEngine) -> None:
    """BLOCKER. The site's input is SymPy's printer output, NOT engine-canonical infix, so
    ``to_prefix`` moves it: ``1/x0`` loses its fitted ``<constant>``, ``x0/2 + x1/2`` loses
    one of two, and ``u ** 0.5`` becomes ``rootn`` -- whose masked index is a constant the
    optimizer cannot fit (nan almost everywhere). Measured 44 of 400 real outputs.
    """
    for expression, old_expected, new_expected in SYMPY_OUTPUTS_DIVERGING:
        old_tokens = mask_all_literals(engine, old_parse(engine, expression))
        new_tokens = mask_all_literals(engine, engine.to_prefix(expression))
        assert old_tokens == old_expected, expression
        assert new_tokens == new_expected, expression
        assert old_tokens != new_tokens


@pytest.mark.xfail(strict=True, reason='BLOCKED: to_prefix moves SymPy output; this is the '
                                       'acceptance criterion for migrating the sympy branch')
def test_model_sympy_branch_is_byte_identical() -> None:
    """The acceptance criterion for migrating ``flash_ansr_model.py``'s ``simplify=='sympy'``
    branch: the beam candidate it emits must not change."""
    engine_ = SimpliPyEngine.load('acj-4-3', install=True)
    for expression, _, _ in SYMPY_OUTPUTS_DIVERGING:
        assert (mask_all_literals(engine_, old_parse(engine_, expression))
                == mask_all_literals(engine_, engine_.to_prefix(expression)))
