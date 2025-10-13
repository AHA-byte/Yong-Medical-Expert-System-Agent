# DxGPT Medical Expert System

This project is a medical expert system agent designed to provide differential diagnosis suggestions based on user-provided symptoms. It leverages AI models and external APIs to deliver concise, evidence-based results for clinical triage.

## Features
- Differential diagnosis for 4–10 symptoms
- Probability-based ranking of diagnoses
- Rationale and red flags for urgent care
- Secure API key management via `.env` file
- Integration with OpenAI and DXGPT APIs

## Setup
1. **Clone the repository:**
   ```powershell
   git clone https://github.com/AHA-byte/Yong-Medical-Expert-System-Agent.git
   cd Yong-Medical-Expert-System-Agent
   ```
2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
3. **Create a `.env` file:**
   Add your API keys and endpoints:
   ```env
   DXGPT_SUBSCRIPTION_KEY=your_dxgpt_subscription_key
   DXGPT_BASE_URL=https://dxgpt-apim.azure-api.net/api
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage
Run the main script for DXGPT-based diagnosis:
```powershell
python DxGPT_Medical_Expert_System.py
```
Or use `OpenAI_Medical_Expert_System.py` for OpenAI-based diagnosis:
```powershell
python OpenAI_Medical_Expert_System.py
```

## Files
* `Dxgpt.py` / `DxGPT_Medical_Expert_System.py`: Main diagnosis logic using DXGPT API
* `OpenAI_Medical_Expert_System.py`: Example usage with OpenAI API
* `.env`: Environment variables for API keys

## Security
- **Never commit your `.env` file with real API keys to Git!**
- Use environment variables to keep credentials safe.

## License
This project is licensed under the MIT License.


