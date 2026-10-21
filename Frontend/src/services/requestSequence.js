export function createRequestSequence() {
  let current = 0;

  return {
    begin() {
      current += 1;
      return current;
    },
    invalidate() {
      current += 1;
    },
    isCurrent(sequence) {
      return sequence === current;
    },
  };
}
