{
  description = "TMDBot env flake";

  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils}:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
        };
      };

      myPyPackages = python-packages: with python-packages; [
        requests
        python-telegram-bot
        autopep8 # autoformatter
        pyyaml
        (buildPythonPackage rec {
          pname = "tmdbv3api";
          version = "1.9.0";

          src = pkgs.fetchPypi {
            inherit pname version;
            sha256 = "sha256-UExdprmcRRb/FgoBV2ES0JfyCcBTT5Q8FcS1bL2Swzs=";
          };

          propagatedBuildInputs = [
            requests
            urllib3
          ];

          doCheck = false;

          pythonImportsCheck = [ "tmdbv3api" ];
        })
      ];

      tmdbot = pkgs.callPackage ({ buildPythonPackage, python-packages }: with python-packages; buildPythonPackage {
          pname = "tmdbot";
          version = "0.1";
          src = ./.;
          propagatedBuildInputs = myPyPackages;
          doCheck = false;
      }) { python-packages = pkgs.python3.pkgs; };

      myPythonWithPackages = pkgs.python3.withPackages myPyPackages;
    in {
      packages = {
        inherit tmdbot;
        default = tmdbot;
      };
      # TODO use stdEnv?
      devShell = pkgs.mkShellNoCC {
        nativeBuildInputs = [
          # Python env for the utility scripts & cocotb
          myPythonWithPackages
        ];
      };
    }
  );
}
