#include <unicode/brkiter.h>
#include <unicode/locid.h>
#include <unicode/regex.h>
#include "unicode_expand.h"

namespace tensorflow
{
namespace miss
{

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
      UErrorCode regexError = U_ZERO_ERROR;

      // Regex for rule 6 and 7
      // ($ALetterEx | $Hebrew_LetterEx) ($MidLetterEx | $MidNumLetEx | $Single_QuoteEx) ($ALetterEx | $Hebrew_LetterEx)
      _wb6 = RegexPattern::compile(
          "([[\\p{Word_Break = ALetter}][\\p{Word_Break = Hebrew_Letter}]]"
          "[[\\p{Word_Break = Extend}][\\p{Word_Break = Format}][\\p{Word_Break = ZWJ}]]*)"
          "[[\\p{Word_Break = MidLetter}][\\p{Word_Break = MidNumLet}][\\p{Word_Break = Single_Quote}]]",
          0,
          regexError);
      OP_REQUIRES(ctx, U_SUCCESS(regexError), errors::InvalidArgument("RegexPattern compilation failed"));

      _wb7 = RegexPattern::compile(
          "([[\\p{Word_Break = MidLetter}][\\p{Word_Break = MidNumLet}][\\p{Word_Break = Single_Quote}]]"
          "[[\\p{Word_Break = Extend}][\\p{Word_Break = Format}][\\p{Word_Break = ZWJ}]]*)"
          "[[\\p{Word_Break = ALetter}][\\p{Word_Break = Hebrew_Letter}]]",
          0,
          regexError);
      OP_REQUIRES(ctx, U_SUCCESS(regexError), errors::InvalidArgument("RegexPattern compilation failed"));

      // Regex for rule 9 and 10
      // $NumericEx ($ALetterEx | $Hebrew_LetterEx) $NumericEx
      _wb9 = RegexPattern::compile(
          "([[\\p{Word_Break = ALetter}][\\p{Word_Break = Hebrew_Letter}]]"
          "[[\\p{Word_Break = Extend}][\\p{Word_Break = Format}][\\p{Word_Break = ZWJ}]]*)"
          "[[\\p{Word_Break = Numeric}][\uFF10-\uff19]]",
          0,
          regexError);
      OP_REQUIRES(ctx, U_SUCCESS(regexError), errors::InvalidArgument("RegexPattern compilation failed"));

      _wb10 = RegexPattern::compile(
          "([[\\p{Word_Break = Numeric}][\uFF10-\uff19]]"
          "[[\\p{Word_Break = Extend}][\\p{Word_Break = Format}][\\p{Word_Break = ZWJ}]]*)"
          "[[\\p{Word_Break = ALetter}][\\p{Word_Break = Hebrew_Letter}]]",
          0,
          regexError);
      OP_REQUIRES(ctx, U_SUCCESS(regexError), errors::InvalidArgument("RegexPattern compilation failed"));

      // Regex for rule 11 and 12
      // $NumericEx ($MidNumEx | $MidNumLetEx | $Single_QuoteEx) $NumericEx
      _wb11 = RegexPattern::compile(
          "([[\\p{Word_Break = Numeric}][\uFF10-\uff19]]"
          "[[\\p{Word_Break = Extend}][\\p{Word_Break = Format}][\\p{Word_Break = ZWJ}]]*)"
          "[[\\p{Word_Break = MidNum}][\\p{Word_Break = MidNumLet}][\\p{Word_Break = Single_Quote}]]",
          0,
          regexError);
      OP_REQUIRES(ctx, U_SUCCESS(regexError), errors::InvalidArgument("RegexPattern compilation failed"));

      _wb12 = RegexPattern::compile(
          "([[\\p{Word_Break = MidNum}][\\p{Word_Break = MidNumLet}][\\p{Word_Break = Single_Quote}]]"
          "[[\\p{Word_Break = Extend}][\\p{Word_Break = Format}][\\p{Word_Break = ZWJ}]]*)"
          "[[\\p{Word_Break = Numeric}][\uFF10-\uff19]]",
          0,
          regexError);
      OP_REQUIRES(ctx, U_SUCCESS(regexError), errors::InvalidArgument("RegexPattern compilation failed"));

      _wbExtSp0 = RegexPattern::compile(
          "([[\\p{Word_Break = ALetter}][\\p{Word_Break = Hebrew_Letter}]"
          "[\\p{Word_Break = Numeric}][\\p{Word_Break = Katakana}]])[\u202F\u2060\u2061\uFEFF]",
          0,
          regexError);
      _wbExtSp1 = RegexPattern::compile(
          "([\u202F\u2060\u2061\uFEFF])[[\\p{Word_Break = ALetter}][\\p{Word_Break = Hebrew_Letter}]"
          "[\\p{Word_Break = Numeric}][\\p{Word_Break = Katakana}]]",
          0,
          regexError);
      OP_REQUIRES(ctx, U_SUCCESS(regexError), errors::InvalidArgument("RegexPattern compilation failed"));
    }
  }

  ~SplitWordsOp()
  {
    delete _wordIterator;
  }

private:
  bool _extended;
  const BreakIterator *_wordIterator;
  const RegexPattern *_wb6;
  const RegexPattern *_wb7;
  const RegexPattern *_wb9;
  const RegexPattern *_wb10;
  const RegexPattern *_wb11;
  const RegexPattern *_wb12;
  const RegexPattern *_wbExtSp0;
  const RegexPattern *_wbExtSp1;

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

      RegexMatcher *matcher6 = _wb6->matcher(unicode_string, extendedError);
      if (!U_SUCCESS(extendedError))
      {
        delete matcher6;

        return false;
      }
      while (matcher6->find(extendedError) && U_SUCCESS(extendedError))
      {
        int32_t end6 = matcher6->end(1, extendedError);
        split_positions.push_back(end6);
        if (!U_SUCCESS(extendedError))
        {
          delete matcher6;

          return false;
        }
      }
      delete matcher6;

      RegexMatcher *matcher7 = _wb7->matcher(unicode_string, extendedError);
      if (!U_SUCCESS(extendedError))
      {
        delete matcher7;

        return false;
      }
      while (matcher7->find(extendedError) && U_SUCCESS(extendedError))
      {
        int32_t end7 = matcher7->end(1, extendedError);
        split_positions.push_back(end7);
        if (!U_SUCCESS(extendedError))
        {
          delete matcher7;

          return false;
        }
      }
      delete matcher7;

      RegexMatcher *matcher9 = _wb9->matcher(unicode_string, extendedError);
      if (!U_SUCCESS(extendedError))
      {
        delete matcher9;

        return false;
      }
      while (matcher9->find(extendedError) && U_SUCCESS(extendedError))
      {
        int32_t end9 = matcher9->end(1, extendedError);
        split_positions.push_back(end9);
        if (!U_SUCCESS(extendedError))
        {
          delete matcher9;

          return false;
        }
      }
      delete matcher9;

      RegexMatcher *matcher10 = _wb10->matcher(unicode_string, extendedError);
      if (!U_SUCCESS(extendedError))
      {
        delete matcher10;

        return false;
      }
      while (matcher10->find(extendedError) && U_SUCCESS(extendedError))
      {
        int32_t end10 = matcher10->end(1, extendedError);
        split_positions.push_back(end10);
        if (!U_SUCCESS(extendedError))
        {
          delete matcher10;

          return false;
        }
      }
      delete matcher10;

      RegexMatcher *matcher11 = _wb11->matcher(unicode_string, extendedError);
      if (!U_SUCCESS(extendedError))
      {
        delete matcher11;

        return false;
      }
      while (matcher11->find(extendedError) && U_SUCCESS(extendedError))
      {
        int32_t end11 = matcher11->end(1, extendedError);
        split_positions.push_back(end11);
        if (!U_SUCCESS(extendedError))
        {
          delete matcher11;

          return false;
        }
      }
      delete matcher11;

      RegexMatcher *matcher12 = _wb12->matcher(unicode_string, extendedError);
      if (!U_SUCCESS(extendedError))
      {
        delete matcher12;

        return false;
      }
      while (matcher12->find(extendedError) && U_SUCCESS(extendedError))
      {
        int32_t end12 = matcher12->end(1, extendedError);
        split_positions.push_back(end12);
        if (!U_SUCCESS(extendedError))
        {
          delete matcher12;

          return false;
        }
      }
      delete matcher12;

      RegexMatcher *matcherExtSp0 = _wbExtSp0->matcher(unicode_string, extendedError);
      if (!U_SUCCESS(extendedError))
      {
        delete matcherExtSp0;

        return false;
      }
      while (matcherExtSp0->find(extendedError) && U_SUCCESS(extendedError))
      {
        int32_t endExtSp0 = matcherExtSp0->end(1, extendedError);
        split_positions.push_back(endExtSp0);
        if (!U_SUCCESS(extendedError))
        {
          delete matcherExtSp0;

          return false;
        }
      }
      delete matcherExtSp0;

      RegexMatcher *matcherExtSp1 = _wbExtSp1->matcher(unicode_string, extendedError);
      if (!U_SUCCESS(extendedError))
      {
        delete matcherExtSp1;

        return false;
      }
      while (matcherExtSp1->find(extendedError) && U_SUCCESS(extendedError))
      {
        int32_t endExtSp1 = matcherExtSp1->end(1, extendedError);
        split_positions.push_back(endExtSp1);
        if (!U_SUCCESS(extendedError))
        {
          delete matcherExtSp1;

          return false;
        }
      }
      delete matcherExtSp1;
    }

    // Remove duplicates and split
    sort(split_positions.begin(), split_positions.end());
    split_positions.erase(unique(split_positions.begin(), split_positions.end()), split_positions.end());

    for (uint64 i = 0; i < split_positions.size() - 1; i++)
    {
      int32_t prev = split_positions[i];
      int32_t pos = split_positions[i + 1];

      if (prev < 0)
        continue;

      UnicodeString word = UnicodeString(unicode_string, prev, pos - prev);

      expanded_strings.push_back(word);
    }

    return true;
  }
};

REGISTER_KERNEL_BUILDER(Name("Miss>SplitWords").Device(DEVICE_CPU), SplitWordsOp);

} // end namespace miss
} // namespace tensorflow
