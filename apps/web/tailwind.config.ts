import type { Config } from "tailwindcss";

const config: Config = {
    darkMode: ["class"],
    content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
  	extend: {
  		colors: {
  			background: 'hsl(var(--background))',
  			foreground: 'hsl(var(--foreground))',
  			card: {
  				DEFAULT: 'hsl(var(--card))',
  				foreground: 'hsl(var(--card-foreground))'
  			},
  			popover: {
  				DEFAULT: 'hsl(var(--popover))',
  				foreground: 'hsl(var(--popover-foreground))'
  			},
  			primary: {
  				DEFAULT: 'hsl(var(--primary))',
  				foreground: 'hsl(var(--primary-foreground))'
  			},
  			secondary: {
  				DEFAULT: 'hsl(var(--secondary))',
  				foreground: 'hsl(var(--secondary-foreground))'
  			},
  			muted: {
  				DEFAULT: 'hsl(var(--muted))',
  				foreground: 'hsl(var(--muted-foreground))'
  			},
  			accent: {
  				DEFAULT: 'hsl(var(--accent))',
  				foreground: 'hsl(var(--accent-foreground))'
  			},
  			destructive: {
  				DEFAULT: 'hsl(var(--destructive))',
  				foreground: 'hsl(var(--destructive-foreground))'
  			},
  			border: 'hsl(var(--border))',
  			input: 'hsl(var(--input))',
  			ring: 'hsl(var(--ring))',
  			chart: {
  				'1': 'hsl(var(--chart-1))',
  				'2': 'hsl(var(--chart-2))',
  				'3': 'hsl(var(--chart-3))',
  				'4': 'hsl(var(--chart-4))',
  				'5': 'hsl(var(--chart-5))'
  			},
  			status: {
  				open: 'hsl(var(--status-open))',
  				acknowledged: 'hsl(var(--status-acknowledged))',
  				dismissed: 'hsl(var(--status-dismissed))',
  				escalated: 'hsl(var(--status-escalated))'
  			},
  			severity: {
  				1: 'hsl(var(--severity-1))',
  				2: 'hsl(var(--severity-2))',
  				3: 'hsl(var(--severity-3))',
  				4: 'hsl(var(--severity-4))',
  				5: 'hsl(var(--severity-5))'
  			},
  			anchor: {
  				teal: '#5BBFB3',
  				'teal-dark': '#4A9E94',
  				charcoal: '#2D3436',
  				warm: '#F9F7F5',
  				coral: '#E17055',
  				mint: '#00B894',
  				gray: '#B2BEC3',
  				dark: '#1E272E',
  			}
  		},
  		fontFamily: {
  			sans: ['var(--font-family)', 'system-ui', '-apple-system', 'sans-serif'],
  			mono: ['var(--font-mono)', 'monospace'],
  		},
  		borderRadius: {
  			'2xl': 'var(--radius-2xl, 1rem)',
  			lg: 'var(--radius)',
  			md: 'var(--radius-md)',
  			sm: 'var(--radius-sm)',
  			anchor: '8px',
  		},
  		boxShadow: {
  			anchor: '0 2px 8px rgba(0,0,0,0.08)',
  		}
  	}
  },
  plugins: [require("tailwindcss-animate")],
};
export default config;
