#include <unicode/brkiter.h>
#include <unicode/locid.h>
#include "unicode_expand.h"

class SplitWordsOp : public UnicodeExpandOp
{
public:
  explicit SplitWordsOp(OpKernelConstruction *ctx) : UnicodeExpandOp(ctx)
  {
    // Prepare attrs
    OP_REQUIRES_OK(ctx, ctx->GetAttr("extended", &_extended));

    // Create word-level BreakIterator instance
    UErrorCode error = U_ZERO_ERROR;
    _wordIterator = BreakIterator::createWordInstance(Locale::getRoot(), error);
    OP_REQUIRES(ctx, U_SUCCESS(error), errors::InvalidArgument("BreakIterator instantiation failed"));
  }

  ~SplitWordsOp()
  {
    delete _wordIterator;
  }

private:
  bool _extended;
  const BreakIterator *_wordIterator;

  const UChar _full_stop = 46;              // \u002E
  const UChar _one_dot_leader = 8228;       // \u2024
  const UChar _small_full_stop = 65106;     // \uFE52
  const UChar _fullwidth_full_stop = 65294; // \uFF0E

  const UChar _colon = 58; // \u003A
  const UChar _vertical_colon = 65043; // \uFE13
  const UChar _small_colon = 65109; // \uFE55
  const UChar _fullwidth_colon = 65306; // \uFF1A

  bool is_stop_char(UChar c)
  {
    return _full_stop == c || _one_dot_leader == c || _small_full_stop == c || _fullwidth_full_stop == c ||
      _colon == c || _vertical_colon == c || _small_colon == c || _fullwidth_colon == c;
  }

  void expand_unicode_extended(const UnicodeString &unicode_string, std::vector<UnicodeString> &expanded_strings)
  {
    if (unicode_string.length() < 2)
    {
      expanded_strings.push_back(unicode_string);
      return;
    }

    // Split words by stop characters
    int32_t prev = 0;
    for (int32_t pos = 0; pos < unicode_string.length(); pos++)
    {
      if (!is_stop_char(unicode_string[pos]))
        continue;

      if (pos < prev)
        continue;

      if (pos > prev)
      {
        UnicodeString word_before = UnicodeString(unicode_string, prev, pos - prev);
        expanded_strings.push_back(word_before);
      }

      UnicodeString stop_mark = UnicodeString(unicode_string, pos, 1);
      expanded_strings.push_back(stop_mark);

      prev = pos + 1;
    }

    if (unicode_string.length() - prev > 0)
    {
      UnicodeString word_after = UnicodeString(unicode_string, prev, unicode_string.length() - prev);
      expanded_strings.push_back(word_after);
    }
  }

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

    // Split words by Unicode rules
    BreakIterator *wordIterator = _wordIterator->clone();
    wordIterator->setText(unicode_string);

    int32_t prev = wordIterator->first();
    for (int32_t pos = wordIterator->first(); pos != BreakIterator::DONE; pos = wordIterator->next())
    {
      if (prev == pos)
        continue;

      UnicodeString word = UnicodeString(unicode_string, prev, pos - prev);
      if (!_extended)
      {
        expanded_strings.push_back(word);
      }
      else
      {
        expand_unicode_extended(word, expanded_strings);
      }

      prev = pos;
    }

    delete wordIterator;

    return true;
  }
};

REGISTER_KERNEL_BUILDER(Name("SplitWords").Device(DEVICE_CPU), SplitWordsOp);
