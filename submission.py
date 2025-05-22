from openai import OpenAI
import re
import time
from typing import Optional

class Submission:
    def __init__(self, max_retries=3):
        """
        Initialize DeepSeek-Prover-V2 client for API-only testing
        
        Args:
            max_retries: Maximum number of API retry attempts
        """
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="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"
        )
        self.model = "deepseek/deepseek-prover-v2:free"
        self.max_retries = max_retries
        print("🚀 DeepSeek-Prover-V2 client initialized (API-only mode)")
    
    def predict(self, full_instruction: str, content_before: str) -> str:
        """
        Generate a diff patch based on instruction and original code
        
        Args:
            full_instruction: Natural language instruction describing the change
            content_before: Full Lean source file before modification
            
        Returns:
            str: Unified diff format patch
        """
        print(f"📝 Processing instruction: {full_instruction[:100]}...")
        
        # Generate patch using API with retry logic
        patch = self._generate_patch_with_retry(full_instruction, content_before)
        
        if patch:
            # Basic format validation
            if self._is_valid_diff_format(patch):
                print("✅ Generated valid diff format")
            else:
                print("⚠️  Generated diff may not be in correct format")
        else:
            print("❌ Failed to generate patch")
        
        return patch
    
    def _generate_patch_with_retry(self, instruction: str, content_before: str) -> str:
        """Generate patch with retry logic for API failures"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                print(f"🔄 API attempt {attempt + 1}/{self.max_retries}")
                return self._generate_patch(instruction, content_before)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"❌ API attempt {attempt + 1} failed: {e}")
                    print(f"⏳ Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"💥 All {self.max_retries} attempts failed. Last error: {e}")
        
        return ""  # Return empty string if all retries failed
    
    def _generate_patch(self, instruction: str, content_before: str) -> str:
        """Internal method to generate patch via API"""
        prompt = self._create_prompt(instruction, content_before)
        
        response = self.client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://example.com",
                "X-Title": "APE-Bench ICML 2025",
            },
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=4000
        )
        
        raw_output = response.choices[0].message.content
        print("✅ API call successful")
        return self._extract_and_clean_diff(raw_output)
    
    def _create_prompt(self, instruction: str, content_before: str) -> str:
        """Create optimized prompt for DeepSeek-Prover-V2"""
        # Analyze instruction type for better prompting
        instruction_type = self._analyze_instruction_type(instruction)
        print(f"🎯 Detected task type: {instruction_type}")
        
        base_prompt = f"""You are an expert Lean 4 theorem prover working with Mathlib4. You need to modify the given Lean source file according to the instruction.

Instruction: {instruction}
Task Type: {instruction_type}

Current Lean file:
```lean
{content_before}
```

Please generate a unified diff patch that implements the requested changes. The patch should:
1. Use proper unified diff format with @@ headers
2. Make minimal necessary changes
3. Ensure the result compiles with Lean 4 and Mathlib4
4. Follow Mathlib4 naming and style conventions
5. Preserve existing functionality"""

        # Add task-specific guidance
        if instruction_type == "ADD_THEOREM":
            base_prompt += """
6. Use appropriate theorem statement syntax
7. Choose suitable proof tactics (simp, ring, norm_num, etc.)
8. Include proper type annotations
9. Follow Mathlib4 naming conventions"""
        elif instruction_type == "FIX_ERROR":
            base_prompt += """
6. Identify and fix the specific error
7. Make minimal changes to preserve functionality
8. Ensure syntax and type correctness"""
        elif instruction_type == "REFACTOR":
            base_prompt += """
6. Improve code organization while preserving semantics
7. Follow Mathlib4 style guidelines
8. Maintain all existing theorems and definitions"""

        base_prompt += """

Generate ONLY the diff patch in this exact format:
```diff
--- a/filename.lean
+++ b/filename.lean
@@ -start_line,line_count +start_line,line_count @@
 context_line
-removed_line
+added_line
 context_line
```

Diff patch:"""
        
        return base_prompt
    
    def _analyze_instruction_type(self, instruction: str) -> str:
        """Analyze instruction to determine task type"""
        instruction_lower = instruction.lower()
        
        if "add" in instruction_lower and ("theorem" in instruction_lower or "lemma" in instruction_lower):
            return "ADD_THEOREM"
        elif "fix" in instruction_lower or "correct" in instruction_lower or "error" in instruction_lower:
            return "FIX_ERROR"
        elif "refactor" in instruction_lower or "reorganiz" in instruction_lower:
            return "REFACTOR"
        elif "remove" in instruction_lower or "delete" in instruction_lower:
            return "REMOVE"
        elif "import" in instruction_lower:
            return "IMPORT_CHANGE"
        else:
            return "GENERAL_EDIT"
    
    def _extract_and_clean_diff(self, raw_output: str) -> str:
        """Extract and clean diff from model output"""
        print("🔍 Extracting diff from model output...")
        
        # Method 1: Extract from ```diff code block
        diff_pattern = r'```diff\s*(.*?)\s*```'
        match = re.search(diff_pattern, raw_output, re.DOTALL)
        
        if match:
            print("✅ Found diff in code block")
            diff_content = match.group(1).strip()
            return self._clean_diff_format(diff_content)
        
        # Method 2: Look for diff patterns in raw text
        print("🔍 Searching for diff patterns in raw text...")
        lines = raw_output.split('\n')
        diff_lines = []
        found_diff_start = False
        
        for line in lines:
            # Start collecting when we see diff headers
            if line.startswith('---') or line.startswith('+++'):
                found_diff_start = True
                print("✅ Found diff header")
            
            if found_diff_start:
                diff_lines.append(line)
                
            # Stop at empty line after substantial content or next code block
            if found_diff_start and (line.strip() == '' or line.startswith('```')) and len(diff_lines) > 5:
                break
        
        if diff_lines:
            print(f"✅ Extracted {len(diff_lines)} diff lines")
        else:
            print("⚠️  No diff pattern found, returning raw output")
            return raw_output
        
        diff_content = '\n'.join(diff_lines).strip()
        return self._clean_diff_format(diff_content)
    
    def _clean_diff_format(self, diff_content: str) -> str:
        """Clean and standardize diff format"""
        if not diff_content:
            return ""
        
        lines = diff_content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.rstrip()  # Remove trailing whitespace
            if line or (cleaned_lines and not cleaned_lines[-1]):  # Keep meaningful empty lines
                cleaned_lines.append(line)
        
        # Remove trailing empty lines
        while cleaned_lines and not cleaned_lines[-1]:
            cleaned_lines.pop()
        
        return '\n'.join(cleaned_lines)
    
    def _is_valid_diff_format(self, diff_content: str) -> bool:
        """Basic validation of diff format"""
        if not diff_content:
            return False
        
        lines = diff_content.split('\n')
        
        # Check for basic diff structure
        has_file_headers = any(line.startswith('---') for line in lines)
        has_plus_headers = any(line.startswith('+++') for line in lines)
        has_hunk_headers = any(line.startswith('@@') for line in lines)
        
        return has_file_headers and has_plus_headers and has_hunk_headers

# Test function to verify API functionality
def test_api_functionality():
    """Comprehensive test of the API-only submission system"""
    print("🧪 Testing DeepSeek-Prover-V2 API functionality...")
    print("=" * 80)
    
    submission = Submission(max_retries=2)
    
    test_cases = [
        {
            "name": "Simple theorem addition",
            "instruction": "Add a theorem that proves 2 + 2 = 4",
            "content": """-- Basic arithmetic file
import Mathlib.Tactic.Ring

-- Some existing content
theorem existing_theorem : True := trivial
"""
        },
        {
            "name": "Fix syntax error",
            "instruction": "Fix the syntax error in the theorem statement where == should be =",
            "content": """import Mathlib.Tactic.Ring

-- This has a syntax error
theorem broken_theorem : 1 + 1 == 2 := by simp
"""
        },
        {
            "name": "Add import statement",
            "instruction": "Add import for Mathlib.Data.Nat.Basic at the top of the file",
            "content": """-- File without proper imports
theorem test_nat : ℕ := 42
"""
        },
        {
            "name": "Complex theorem addition",
            "instruction": "Add a theorem that proves the commutativity of addition: a + b = b + a",
            "content": """import Mathlib.Tactic.Ring
import Mathlib.Algebra.Ring.Basic

-- Existing content
theorem simple_theorem : 1 = 1 := rfl
"""
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"🧪 Test {i}/{len(test_cases)}: {test_case['name']}")
        print(f"📋 Instruction: {test_case['instruction']}")
        print("=" * 80)
        
        try:
            start_time = time.time()
            result = submission.predict(test_case['instruction'], test_case['content'])
            end_time = time.time()
            
            print(f"⏱️  Time taken: {end_time - start_time:.2f} seconds")
            
            if result:
                print("\n📄 Generated diff:")
                print("-" * 40)
                print(result)
                print("-" * 40)
                
                # Basic format validation
                if submission._is_valid_diff_format(result):
                    print("✅ Valid diff format detected")
                    results.append(True)
                else:
                    print("⚠️  Diff format validation failed")
                    results.append(False)
            else:
                print("❌ No diff generated")
                results.append(False)
                
        except Exception as e:
            print(f"💥 Test failed with error: {e}")
            results.append(False)
        
        # Add delay between tests to be respectful to API
        if i < len(test_cases):
            print("⏳ Waiting 3 seconds before next test...")
            time.sleep(3)
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 TEST SUMMARY")
    print("=" * 80)
    passed = sum(results)
    total = len(results)
    success_rate = (passed / total) * 100
    
    print(f"✅ Passed: {passed}/{total} ({success_rate:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! API integration is working perfectly.")
        print("🚀 Ready for competition submission!")
    elif passed > 0:
        print("⚠️  Some tests passed. API is working but may need refinement.")
        print("🔧 Consider improving prompts or error handling.")
    else:
        print("❌ All tests failed. Check API credentials and connectivity.")
    
    print("\n💡 Next steps:")
    if passed > 0:
        print("1. Create the final submission zip file")
        print("2. Test with more complex examples")
        print("3. Submit to the competition platform")
    else:
        print("1. Check API key and network connectivity")
        print("2. Verify OpenRouter service status")
        print("3. Check error messages above")
    
    return results

def quick_test():
    """Quick single test for immediate feedback"""
    print("⚡ Quick API test...")
    submission = Submission(max_retries=1)
    
    simple_instruction = "Add a simple theorem that 1 + 1 = 2"
    simple_content = "import Mathlib.Tactic.Ring\n\n-- Add theorem here"
    
    result = submission.predict(simple_instruction, simple_content)
    
    if result:
        print("✅ Quick test successful!")
        print(f"Generated: {result[:100]}...")
        return True
    else:
        print("❌ Quick test failed")
        return False

if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_test()
    else:
        test_api_functionality()
