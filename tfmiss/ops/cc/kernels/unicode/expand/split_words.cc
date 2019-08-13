#include <unicode/brkiter.h>
#include <unicode/locid.h>
#include <unicode/regex.h>
#include "unicode_expand.h"

class SplitWordsOp : public UnicodeExpandOp
{
public:
  explicit SplitWordsOp(OpKernelConstruction *ctx) : UnicodeExpandOp(ctx)
  {
    // Prepare attrs
    OP_REQUIRES_OK(ctx, ctx->GetAttr("extended", &_extended));

    // Create word-level BreakIterator instance
    UErrorCode iteratorError = U_ZERO_ERROR;
    _wordIterator = BreakIterator::createWordInstance(Locale::getRoot(), iteratorError);
    OP_REQUIRES(ctx, U_SUCCESS(iteratorError), errors::InvalidArgument("BreakIterator instantiation failed"));

    if (_extended)
    {
      // Regex for rule 6 and 7
      // ($ALetterEx | $Hebrew_LetterEx) ($MidLetterEx | $MidNumLetEx | $Single_QuoteEx) ($ALetterEx | $Hebrew_LetterEx)
      UErrorCode regex67Error = U_ZERO_ERROR;
      _wb67 = RegexPattern::compile(
          "([[\\p{Word_Break = ALetter}] [\\p{Word_Break = Hebrew_Letter}]])"
          "([[\\p{Word_Break = Extend}] [\\p{Word_Break = Format}] [\\p{Word_Break = ZWJ}]])*"
          "([[\\p{Word_Break = MidLetter}] [\\p{Word_Break = MidNumLet}] [\\p{Word_Break = Single_Quote}]])"
          "([[\\p{Word_Break = Extend}] [\\p{Word_Break = Format}] [\\p{Word_Break = ZWJ}]])*"
          "([[\\p{Word_Break = ALetter}] [\\p{Word_Break = Hebrew_Letter}]])",
          0,
          regex67Error);
      OP_REQUIRES(ctx, U_SUCCESS(regex67Error), errors::InvalidArgument("RegexPattern compilation failed"));

      // Regex for rule 11 and 12
      // $NumericEx ($MidNumEx | $MidNumLetEx | $Single_QuoteEx) $NumericEx
      UErrorCode regex1112Error = U_ZERO_ERROR;
      _wb1112 = RegexPattern::compile(
          "([[\\p{Word_Break = Numeric}] [\uFF10-\uff19]])"
          "([[\\p{Word_Break = Extend}] [\\p{Word_Break = Format}] [\\p{Word_Break = ZWJ}]])*"
          "([[\\p{Word_Break = MidNum}] [\\p{Word_Break = MidNumLet}] [\\p{Word_Break = Single_Quote}]])"
          "([[\\p{Word_Break = Extend}] [\\p{Word_Break = Format}] [\\p{Word_Break = ZWJ}]])*"
          "([[\\p{Word_Break = Numeric}] [\uFF10-\uff19]])",
          0,
          regex1112Error);
      OP_REQUIRES(ctx, U_SUCCESS(regex1112Error), errors::InvalidArgument("RegexPattern compilation failed"));
    }
  }

  ~SplitWordsOp()
  {
    delete _wordIterator;
  }

private:
  bool _extended;
  const BreakIterator *_wordIterator;
  const RegexPattern *_wb67;
  const RegexPattern *_wb1112;

protected:
  uint64 expand_rate() override
  {
    return (uint64)2; // Hard to say, but usualy it will split something
  }

  bool expand_unicode(const UnicodeString &unicode_string, std::vector<UnicodeString> &expanded_strings) override
  {
    if (unicode_string.length() < 2)
    {
      expanded_strings.push_back(unicode_string);
      return true;
    }

    std::vector<int32_t> split_positions;

    // Split words by Unicode rules
    BreakIterator *wordIterator = _wordIterator->clone();
    wordIterator->setText(unicode_string);
    for (int32_t pos = wordIterator->first(); pos != BreakIterator::DONE; pos = wordIterator->next())
    {
      split_positions.push_back(pos);
    }
    delete wordIterator;

    // Split words ignoring WB 6, 7, 11 and 12
    if (_extended)
    {
      UErrorCode extendedError = U_ZERO_ERROR;

      RegexMatcher *matcher67 = _wb67->matcher(unicode_string, extendedError);
      if (!U_SUCCESS(extendedError))
      {
        delete matcher67;

        return false;
      }

      while (matcher67->find(extendedError) && U_SUCCESS(extendedError))
      {
        for (int i = 1; i < 5; i++)
        {
          int32_t end67 = matcher67->end(i, extendedError);
          if (end67 >= 0)
            split_positions.push_back(end67);
          if (!U_SUCCESS(extendedError))
          {
            delete matcher67;

            return false;
          }
        }
      }

      delete matcher67;

      RegexMatcher *matcher1112 = _wb1112->matcher(unicode_string, extendedError);
      if (!U_SUCCESS(extendedError))
      {
        delete matcher1112;

        return false;
      }

      while (matcher1112->find(extendedError) && U_SUCCESS(extendedError))
      {
        for (int i = 1; i < 5; i++)
        {
          int32_t end1112 = matcher1112->end(i, extendedError);
          if (end1112 >= 0)
            split_positions.push_back(end1112);
          if (!U_SUCCESS(extendedError))
          {
            delete matcher1112;

            return false;
          }
        }
      }

      delete matcher1112;
    }

    // Remove duplicates and split
    sort(split_positions.begin(), split_positions.end());
    split_positions.erase(unique(split_positions.begin(), split_positions.end()), split_positions.end());

    for (uint64 i = 0; i < split_positions.size() - 1; i++)
    {
      int32_t prev = split_positions[i];
      int32_t pos = split_positions[i + 1];

      UnicodeString word = UnicodeString(unicode_string, prev, pos - prev);

      expanded_strings.push_back(word);
    }

    return true;
  }
};

REGISTER_KERNEL_BUILDER(Name("SplitWords").Device(DEVICE_CPU), SplitWordsOp);
