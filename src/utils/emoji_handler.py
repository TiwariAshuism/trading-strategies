#!/usr/bin/env python3
"""
Emoji Handling Utility
Provides safe emoji printing with ASCII fallbacks for all system components.
"""

import sys
import os

class EmojiHandler:
    """Handles emoji display with fallbacks for different terminal encodings"""
    
    # Emoji mappings to ASCII alternatives
    EMOJI_MAP = {
        # System and status emojis
        '🚀': '[LAUNCH]',
        '✅': '[OK]',
        '': '[ERROR]',
        '⚠️': '[WARNING]',
        '💡': '[TIP]',
        '🔧': '[TOOL]',
        '🔍': '[INFO]',
        '📊': '[CHART]',
        '📈': '[UP]',
        '📉': '[DOWN]',
        '💰': '[MONEY]',
        '💾': '[DATA]',
        '🌐': '[WEB]',
        '📱': '[MOBILE]',
        '💻': '[COMPUTER]',
        
        # Trading specific emojis
        '📋': '[LIST]',
        '🎯': '[TARGET]',
        '⚡': '[FAST]',
        '🔄': '[REFRESH]',
        '🎪': '[ANALYSIS]',
        '🏆': '[WIN]',
        '📰': '[NEWS]',
        '🤖': '[BOT]',
        '📡': '[FEED]',
        '🔮': '[PREDICT]',
        '⏰': '[TIME]',
        '🎲': '[RANDOM]',
        '🧠': '[BRAIN]',
        '🛑': '[STOP]',
        '🔐': '[SECURE]',
        
        # Numbers and symbols
        '1️⃣': '[1]',
        '2️⃣': '[2]',
        '3️⃣': '[3]',
        '4️⃣': '[4]',
        '5️⃣': '[5]',
        '🟢': '[GREEN]',
        '🔴': '[RED]',
        '🟡': '[YELLOW]',
        '⬆️': '[UP]',
        '⬇️': '[DOWN]',
        '➡️': '[RIGHT]',
        '🔥': '[HOT]',
        '❄️': '[COLD]',
        
        # Status indicators
        '🎉': '[SUCCESS]',
        '💪': '[STRONG]',
        '🚨': '[ALERT]',
        '📣': '[ANNOUNCE]',
        '🔔': '[NOTIFY]',
        '💯': '[100]',
        '🎨': '[DESIGN]',
        '🛠️': '[BUILD]',
        '⭐': '[STAR]',
        '🏁': '[FINISH]',
        '🔹': '[BULLET]',
        '▶️': '[PLAY]',
        '⏸️': '[PAUSE]',
        '⏹️': '[STOP]',
        
        # File and folder emojis
        '📁': '[FOLDER]',
        '📂': '[OPEN_FOLDER]',
        '📄': '[FILE]',
        '📝': '[EDIT]',
        '💼': '[PORTFOLIO]',
        '🗂️': '[ORGANIZE]',
        '🏷️': '[TAG]',
        '📌': '[PIN]',
        '🔗': '[LINK]',
        '📎': '[ATTACH]',
        
        # Activity emojis
        '🏃': '[RUN]',
        '🔊': '[SOUND]',
        '🔇': '[MUTE]',
        '🔆': '[BRIGHT]',
        '🔅': '[DIM]',
        '🔋': '[BATTERY]',
        '⚙️': '[SETTINGS]',
        '🔄': '[SYNC]',
        '🔃': '[CYCLE]',
        '🔀': '[SHUFFLE]',
        
        # Testing emojis
        '🧪': '[TEST]',
        '🔬': '[ANALYZE]',
        '📏': '[MEASURE]',
        '📐': '[CALCULATE]',
        '🧮': '[COMPUTE]',
        '📊': '[STATS]',
        '📈': '[GROWTH]',
        '📉': '[DECLINE]',
        
        # Special characters
        '═': '=',
        '║': '|',
        '╔': '+',
        '╗': '+',
        '╚': '+',
        '╝': '+',
        '╠': '+',
        '╣': '+',
        '╦': '+',
        '╩': '+',
        '╬': '+',
    }
    
    @classmethod
    def supports_unicode(cls):
        """Check if the current terminal supports Unicode"""
        try:
            # Check encoding
            encoding = sys.stdout.encoding or 'ascii'
            if encoding.lower() in ['utf-8', 'utf8']:
                return True
            
            # Check environment variables
            lang = os.environ.get('LANG', '').lower()
            if 'utf-8' in lang or 'utf8' in lang:
                return True
            
            # Test print
            test_emoji = '✅'
            sys.stdout.write(test_emoji)
            sys.stdout.flush()
            return True
            
        except (UnicodeEncodeError, UnicodeError):
            return False
        except Exception:
            return False
    
    @classmethod
    def safe_print(cls, text, end='\n', flush=False):
        """Print text with emoji fallback handling"""
        try:
            print(text, end=end, flush=flush)
        except UnicodeEncodeError:
            # Replace emojis with ASCII alternatives
            safe_text = cls.replace_emojis(text)
            #print(safe_text, end=end, flush=flush)
    
    @classmethod
    def replace_emojis(cls, text):
        """Replace emojis in text with ASCII alternatives"""
        if not isinstance(text, str):
            text = str(text)
        
        for emoji, replacement in cls.EMOJI_MAP.items():
            text = text.replace(emoji, replacement)
        
        return text
    
    @classmethod
    def format_text(cls, text):
        """Format text based on terminal capabilities"""
        if cls.supports_unicode():
            return text
        else:
            return cls.replace_emojis(text)

# Convenience functions for easy import
def safe_print(text, end='\n', flush=False):
    """Safe print function with emoji handling"""
    EmojiHandler.safe_print(text, end=end, flush=flush)

def format_text(text):
    """Format text with emoji handling"""
    return EmojiHandler.format_text(text)

def replace_emojis(text):
    """Replace emojis with ASCII alternatives"""
    return EmojiHandler.replace_emojis(text)

# Test function
def test_emoji_handling():
    """Test emoji handling capabilities"""
    test_texts = [
        "🚀 System starting...",
        "✅ All tests passed!",
        " Error occurred",
        "📊 Portfolio Analysis",
        "🎯 Trading Signals",
        "💰 Profit: $1000",
        "⚠️ Warning: High volatility"
    ]
    
    print("Testing emoji handling:")
    print("-" * 30)
    
    for text in test_texts:
        safe_print(f"Original: {text}")
        safe_print(f"Safe:     {replace_emojis(text)}")
        print()
    
    print(f"Unicode support: {EmojiHandler.supports_unicode()}")

if __name__ == "__main__":
    test_emoji_handling()